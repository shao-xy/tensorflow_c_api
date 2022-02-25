#include <cstdlib>
#include <random>

#include "TestData.h"
#include "TFContext.h"

#define SAFE_FREE(p) \
	do { \
		if (p) free(p); \
		p = 0; \
	} while (0)

TestData::TestData(TFContext * ctx)
	: ctx(ctx)
{
	init_output(&input, &num_inputs, NUM_INPUTS, "serving_default_input_");
	init_output(&output, &num_outputs, NUM_OUTPUTS, "StatefulPartitionedCall", false);

	init_data(&dims, &ndims, &data, &ndata);

	init_tensors();
}

TestData::~TestData()
{
	ctx = 0;

	SAFE_FREE(input);
	SAFE_FREE(output);
	SAFE_FREE(input_vals);
	SAFE_FREE(output_vals);
	SAFE_FREE(dims);
	SAFE_FREE(data);
}

bool TestData::init_output(TF_Output ** p_outptr, int * p_num, int size, string op_name_prefix, bool mark_id)
{
	(*p_num) = size;
	(*p_outptr) = (TF_Output *) malloc(sizeof(TF_Output) * size);
	for (int i = 0; i < size; i++) {
		string graph_op_name = op_name_prefix;
		if (mark_id) {
			graph_op_name += std::to_string(i+1);
		}
		TF_Output element = {TF_GraphOperationByName(ctx->graph, graph_op_name.c_str()), 0};
		if (!element.oper)	return false;
		(*p_outptr)[i] = element;
	}
	return true;
}

void TestData::init_data(int64_t ** p_dims, int * p_ndims, float ** p_data, int * p_ndata)
{
	(*p_ndims) = 3;
	(*p_dims) = (int64_t *) malloc((*p_ndims) * sizeof(int64_t));
	(*p_dims)[0] = 1;
	(*p_dims)[1] = 48;
	(*p_dims)[2] = 2258;

	int64_t total_data_size = 1;
	for (int i = 0; i < (*p_ndims); i++) {
		total_data_size *= (*p_dims)[i];
	}
	(*p_ndata) = total_data_size * sizeof(float);

	static std::default_random_engine e;
	static std::uniform_real_distribution<> dis(0, 1);

	(*p_data) = (float *) malloc(*p_ndata);
	for (int i = 0; i < total_data_size; i++) {
		//(*p_data)[i] = 0.8;
		(*p_data)[i] = dis(e);
	}
}

bool TestData::init_tensors()
{
	input_vals = (TF_Tensor **) malloc(num_inputs * sizeof(TF_Tensor *));
	output_vals = (TF_Tensor **) malloc(num_outputs * sizeof(TF_Tensor *));

	TF_Tensor * input_tensor = TF_NewTensor(TF_FLOAT, dims, ndims, data, ndata, &NoOpDeallocator, 0);
	if (!input_tensor) {
		return false;
	}
	input_vals[0] = input_tensor;
	return true;
}
