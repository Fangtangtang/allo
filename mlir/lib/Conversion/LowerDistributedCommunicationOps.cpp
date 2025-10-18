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

bool applyLowerDistributedCommunicationOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    std::string funcName = func.getName().str();
    llvm::outs() << "Function name: " << funcName << "\n";
    SmallVector<GatherOp, 8> setGatherOps;
    SmallVector<ScatterOp, 8> setScatterOps;
    func.walk([&](Operation *op) {
      if (auto gatherOp = dyn_cast<GatherOp>(op)) {
        setGatherOps.push_back(gatherOp);
      }
      if (auto scatterOp = dyn_cast<ScatterOp>(op)) {
        setScatterOps.push_back(scatterOp);
      }
    });
    // TODO: lowering
    for (size_t i = 0; i < setGatherOps.size(); ++i) {
      auto op = setGatherOps[i];
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      auto namesAttr = op->getAttr("names");
      auto number = namesAttr.cast<ArrayAttr>().size();
      Value stream = op->getOperands()[0];
      auto streamBaseType = stream.getType()
                                .dyn_cast<StreamType>()
                                .getBaseType()
                                .dyn_cast<MemRefType>();
      Value output = op->getResults()[0];
      // allocate a local buffer
      Value buffer = rewriter.create<memref::AllocOp>(
          loc, output.getType().dyn_cast<MemRefType>());
      // single nest loop (get from stream, copy to slice of the buffer)
      Value lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
      Value ub = rewriter.create<arith::ConstantIndexOp>(loc, number);
      Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);
      auto forOp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
      // attributes for the loop
      std::string opName = "S_" + funcName + "_" + std::to_string(i);
      std::string iterName = funcName + std::to_string(i);
      forOp->setAttr("op_name", StringAttr::get(op->getContext(), opName));
      forOp->setAttr("loop_name", StringAttr::get(op->getContext(), iterName));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
      // get op
      auto get_op = bodyBuilder.create<StreamGetOp>(
          loc, streamBaseType, stream, bodyBuilder.getDenseI64ArrayAttr({}));
      // subview
      SmallVector<Value> offsetValues, emptyValues;
      SmallVector<int64_t> staticOffsets, staticSizes, staticStrides, strides;
      auto memref_shape = output.getType().dyn_cast<MemRefType>().getShape();
      int64_t tileSize = 1;
      for (int64_t dim : memref_shape) {
        tileSize *= dim;
      }
      for (int64_t i = 0; i < memref_shape.size(); ++i) {
        tileSize /= memref_shape[i];
        staticStrides.push_back(1);
        if (i == 0) {
          offsetValues.push_back(forOp.getInductionVar());
          staticOffsets.push_back(ShapedType::kDynamic);
          staticSizes.push_back(1);
        } else {
          staticOffsets.push_back(0);
          staticSizes.push_back(memref_shape[i]);
          strides.push_back(tileSize);
        }
      }
      auto layout = StridedLayoutAttr::get(bodyBuilder.getContext(),
                                           ShapedType::kDynamic, strides);
      auto subViewType = MemRefType::get(
          streamBaseType.getShape(), streamBaseType.getElementType(), layout);
      auto subview_op = bodyBuilder.create<memref::SubViewOp>(
          loc, subViewType, buffer,
          offsetValues,
          emptyValues,
          emptyValues,
          staticOffsets, staticSizes, staticStrides);
      // copy
      bodyBuilder.create<memref::CopyOp>(loc, get_op, subview_op);
      output.replaceAllUsesWith(buffer);
      op->erase();
    }
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerDistributedCommunicationOpsTransformation
    : public LowerDistributedCommunicationOpsBase<
          AlloLowerDistributedCommunicationOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerDistributedCommunicationOps(mod)) {
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
