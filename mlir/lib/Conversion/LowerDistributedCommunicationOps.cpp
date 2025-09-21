/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "allo/Conversion/Passes.h"
#include "allo/Dialect/AlloDialect.h"
#include "allo/Dialect/AlloOps.h"
#include "allo/Dialect/AlloTypes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace allo;

//===----------------------------------------------------------------------===//
// Lower GatherOp Ops
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {
bool applyLowerGatherOps(ModuleOp &mod) {
  // TODO
  return false;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerGatherOpsTransformation
    : public LowerGatherOpsBase<AlloLowerGatherOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerGatherOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Lower ScatterOp Ops
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {
bool applyLowerScatterOps(ModuleOp &mod) {
  // TODO
  return false;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerScatterOpsTransformation
    : public LowerScatterOpsBase<AlloLowerScatterOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerScatterOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>> createLowerGatherOpsPass() {
  return std::make_unique<AlloLowerGatherOpsTransformation>();
}

std::unique_ptr<OperationPass<ModuleOp>> createLowerScatterOpsPass() {
  return std::make_unique<AlloLowerScatterOpsTransformation>();
}
} // namespace allo
} // namespace mlir