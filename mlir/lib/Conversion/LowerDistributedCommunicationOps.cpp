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
// Lower Gather&Scatter Ops
//===----------------------------------------------------------------------===//
namespace mlir {
namespace allo {
bool applyLowerGatherOps(ModuleOp &mod) {
  // TODO
  return false;
}

bool applyLowerScatterOps(ModuleOp &mod) {
  // TODO
  return false;
}
} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerDistributedCommunicationOpsTransformation
    : public LowerDistributedCommunicationOpsBase<
          AlloLowerDistributedCommunicationOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerGatherOps(mod)) {
      return signalPassFailure();
    }
    if (!applyLowerScatterOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerDistributedCommunicationOpsPass() {
  return std::make_unique<AlloLowerDistributedCommunicationOpsTransformation>();
}

} // namespace allo
} // namespace mlir
