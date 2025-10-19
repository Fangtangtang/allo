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

struct SubViewParams {
  SmallVector<int64_t> staticOffsets;
  SmallVector<int64_t> staticSizes;
  SmallVector<int64_t> staticStrides;
  SmallVector<int64_t> strides;
};

static scf::ForOp buildLoop(OpBuilder &builder, Location loc,
                            int64_t upperBound, Attribute iterName) {
  Value lb = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value ub = builder.create<arith::ConstantIndexOp>(loc, upperBound);
  Value step = builder.create<arith::ConstantIndexOp>(loc, 1);
  auto forOp = builder.create<scf::ForOp>(loc, lb, ub, step);
  // set attributes
  forOp->setAttr("op_name", iterName);
  forOp->setAttr("loop_name", iterName);
  return forOp;
}

SubViewParams computeSubViewParams(ArrayRef<int64_t> memrefShape) {
  SubViewParams params;
  int64_t tileSize = 1;
  for (int64_t dim : memrefShape) {
    tileSize *= dim;
  }
  for (int64_t i = 0; i < memrefShape.size(); ++i) {
    tileSize /= memrefShape[i];
    params.staticStrides.push_back(1);
    if (i == 0) {
      params.staticOffsets.push_back(ShapedType::kDynamic);
      params.staticSizes.push_back(1);
    } else {
      params.staticOffsets.push_back(0);
      params.staticSizes.push_back(memrefShape[i]);
      params.strides.push_back(tileSize);
    }
  }
  return params;
}

bool applyLowerCollectiveOps(ModuleOp &mod) {
  for (func::FuncOp func : mod.getOps<func::FuncOp>()) {
    std::string funcName = func.getName().str();
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
    for (size_t i = 0; i < setGatherOps.size(); ++i) {
      auto op = setGatherOps[i];
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      auto namesAttr = op->getAttr("names");
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
      auto forOp = buildLoop(rewriter, loc, namesAttr.cast<ArrayAttr>().size(),
                             op->getAttr("iter_name"));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
      // get stream
      auto get_op = bodyBuilder.create<StreamGetOp>(
          loc, streamBaseType, stream, bodyBuilder.getDenseI64ArrayAttr({}));
      // subview
      SmallVector<Value> offsetValues, emptyValues;
      offsetValues.push_back(forOp.getInductionVar());
      auto memrefShape = output.getType().dyn_cast<MemRefType>().getShape();
      auto params = computeSubViewParams(memrefShape);
      auto layout = StridedLayoutAttr::get(
          bodyBuilder.getContext(), ShapedType::kDynamic, params.strides);
      auto subViewType = MemRefType::get(
          streamBaseType.getShape(), streamBaseType.getElementType(), layout);
      auto subview_op = bodyBuilder.create<memref::SubViewOp>(
          loc, subViewType, buffer, offsetValues, emptyValues, emptyValues,
          params.staticOffsets, params.staticSizes, params.staticStrides);
      // copy
      bodyBuilder.create<memref::CopyOp>(loc, get_op, subview_op);

      output.replaceAllUsesWith(buffer);
      op->erase();
    }
    for (size_t i = 0; i < setScatterOps.size(); ++i) {
      auto op = setScatterOps[i];
      Location loc = op->getLoc();
      OpBuilder rewriter(op);
      Value stream = op->getOperands()[0];
      auto streamBaseType = stream.getType()
                                .dyn_cast<StreamType>()
                                .getBaseType()
                                .dyn_cast<MemRefType>();
      Value buffer = op->getOperands()[1];
      auto namesAttr = op->getAttr("names");
      auto forOp = buildLoop(rewriter, loc, namesAttr.cast<ArrayAttr>().size(),
                             op->getAttr("iter_name"));
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(forOp.getBody());
      // subview
      SmallVector<Value> offsetValues, emptyValues;
      offsetValues.push_back(forOp.getInductionVar());
      auto memrefShape = buffer.getType().dyn_cast<MemRefType>().getShape();
      auto params = computeSubViewParams(memrefShape);
      auto layout = StridedLayoutAttr::get(
          bodyBuilder.getContext(), ShapedType::kDynamic, params.strides);
      auto subViewType = MemRefType::get(
          streamBaseType.getShape(), streamBaseType.getElementType(), layout);
      auto subview_op = bodyBuilder.create<memref::SubViewOp>(
          loc, subViewType, buffer, offsetValues, emptyValues, emptyValues,
          params.staticOffsets, params.staticSizes, params.staticStrides);
      // put stream
      bodyBuilder.create<StreamPutOp>(
          loc, stream, bodyBuilder.getDenseI64ArrayAttr({}), subview_op);
      op->erase();
    }
  }
  return true;
}

} // namespace allo
} // namespace mlir

namespace {
struct AlloLowerCollectiveOpsTransformation
    : public LowerCollectiveOpsBase<
          AlloLowerCollectiveOpsTransformation> {
  void runOnOperation() override {
    auto mod = getOperation();
    if (!applyLowerCollectiveOps(mod)) {
      return signalPassFailure();
    }
  }
};
} // namespace

namespace mlir {
namespace allo {

std::unique_ptr<OperationPass<ModuleOp>>
createLowerCollectiveOpsPass() {
  return std::make_unique<AlloLowerCollectiveOpsTransformation>();
}

} // namespace allo
} // namespace mlir
