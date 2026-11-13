"""Feature engineering that lives *inside* the model pipeline, plus its ONNX converter,
so the browser demo computes exactly the same features as training."""
import numpy as np
import onnx
from sklearn.base import BaseEstimator, TransformerMixin
from skl2onnx import update_registered_converter
from skl2onnx.algebra.onnx_ops import OnnxAdd, OnnxCast, OnnxConcat, OnnxDiv, OnnxEqual, OnnxGather, OnnxLog
from skl2onnx.common.data_types import FloatTensorType

FAMILY_FEATURES = ["family_size", "is_alone", "log_fare", "log_fare_per_person"]


class FamilyFeatures(BaseEstimator, TransformerMixin):
    """[SibSp, Parch, Fare] -> [family_size, is_alone, log(1+fare), log(1+fare / family_size)]."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # float32 on purpose: identical arithmetic to the ONNX graph used in the browser
        X = np.asarray(X, dtype=np.float32)
        one = np.float32(1)
        family = X[:, 0] + X[:, 1] + one
        out = np.c_[family, (family == one).astype(np.float32),
                    np.log(X[:, 2] + one), np.log(X[:, 2] / family + one)]
        return out.astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(FAMILY_FEATURES, dtype=object)


def _shape(operator):
    operator.outputs[0].type = FloatTensorType([operator.inputs[0].get_first_dimension(), len(FAMILY_FEATURES)])


def _convert(scope, operator, container):
    v = container.target_opset
    X = operator.inputs[0]
    one = np.array([1], dtype=np.float32)

    def col(i):
        return OnnxGather(X, np.array([i], dtype=np.int64), axis=1, op_version=v)

    family = OnnxAdd(OnnxAdd(col(0), col(1), op_version=v), one, op_version=v)
    alone = OnnxCast(OnnxEqual(family, one, op_version=v), to=onnx.TensorProto.FLOAT, op_version=v)
    log_fare = OnnxLog(OnnxAdd(col(2), one, op_version=v), op_version=v)
    log_pp = OnnxLog(OnnxAdd(OnnxDiv(col(2), family, op_version=v), one, op_version=v), op_version=v)
    OnnxConcat(family, alone, log_fare, log_pp, axis=1, op_version=v,
               output_names=operator.outputs[:1]).add_to(scope, container)


update_registered_converter(FamilyFeatures, "TitanicFamilyFeatures", _shape, _convert)
