#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const SigmoidFineCEParameter  fine_ce_param = this->layer_param_.sigmoid_fine_ce_loss_param();
//  range_ = fine_ce_param.threshold_range();
  alpha_ = fine_ce_param.threshold_alpha();
  beta_  = fine_ce_param.threshold_beta();

// CHECK_GT(range_, 1) << "Number of neighboring pixels to take into account must be greater thatn 1.";
  LOG(INFO) << "This implementation does not support train batch size > 1.";
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count()) <<
      "SIGMOID_FINE_CROSS_ENTROPY_LOSS layer inputs must have the same count.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  //const int count = bottom[0]->count();
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  int dim = bottom[0]->count() / bottom[0]->num();

  for (int j = 0; j < dim; j++) {
	  if (target[j] == 1) {
		  count_pos++;
	  } else if (target[j] == 0){
		  count_neg++;
	  }
  }

  for (int j = 0; j < dim; j++) {
	if (target[j] == 1 && sigmoid_output_data[j] < alpha_) {
	  loss_pos -= (input_data[j]*(target[j]-(input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
	}
	if (target[j] == 0 && sigmoid_output_data[j] > beta_) {
	  loss_neg -= (input_data[j]*(target[j] - (input_data[j]>=0)) - log(1 + exp(input_data[j] - 2*input_data[j]*(input_data[j] >= 0))));
	}
  }
  loss_pos *= count_neg / (count_pos + count_neg);
  loss_neg *= count_pos / (count_pos + count_neg);

  top[0]->mutable_cpu_data()[0] = loss_pos + loss_neg;
}

template <typename Dtype>
void SigmoidFineCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    caffe_sub(count, sigmoid_output_data, target, bottom_diff);
	Dtype count_pos = 0;
	Dtype count_neg = 0;
    int dim = bottom[0]->count() / bottom[0]->num();

    for (int j = 0; j < dim; j ++) {
      if (target[j] == 1) {
       	count_pos ++;
	  } else if (target[j] == 0){
       	count_neg ++;
	  }
    }
    for (int j = 0; j < dim; j ++) {
       	if (target[j] == 1) { /* positive */
			if (sigmoid_output_data[j] < alpha_) {
        		bottom_diff[j] *= 1 * count_neg / (count_pos + count_neg);
        	} else {
        		bottom_diff[j] = 0;
			}
    	} else if (target[j] == 0){ /* negative */
			if (sigmoid_output_data[j] > beta_) {
                bottom_diff[j] *= 1 * count_pos / (count_pos + count_neg);
			} else {
				bottom_diff[j] = 0;
			}
		} else {
			/* ignore none 0 or 1 */
		}
    }

    const Dtype loss_weight = top [0]->cpu_diff()[0];
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU_BACKWARD(SigmoidFineCrossEntropyLossLayer, Backward);
#endif

INSTANTIATE_CLASS(SigmoidFineCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(SigmoidFineCrossEntropyLoss);

}  // namespace caffe
