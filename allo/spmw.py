# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
from pathlib import Path

MODULE_CACHE = {}


def get_module(filename):
    if filename not in MODULE_CACHE:
        src = Path(filename).read_text()
        MODULE_CACHE[filename] = (src, ast.parse(src))
    return MODULE_CACHE[filename]


def find_function_ast(fn):
    filename = fn.__code__.co_filename
    lineno = fn.__code__.co_firstlineno + 1
    src, tree = get_module(filename)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno == lineno:
            return src, node
    raise RuntimeError("Function AST not found")


def kernel(fn):
    src, node = find_function_ast(fn)
    fn._source = ast.get_source_segment(src, node)
    fn._ast = node
    return fn


def unit():
    return kernel


def work(*, mapping: list[int], args=None):
    def decorator(fn):
        src, node = find_function_ast(fn)
        fn._df_meta = {
            "mapping": mapping,
            "args": [] if args is None else args,
        }
        fn._source = ast.get_source_segment(src, node)
        fn._ast = node

        return fn

    return decorator
