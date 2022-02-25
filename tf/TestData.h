#ifndef _TF_TESTDATA_H_
#define _TF_TESTDATA_H_

#include <string>
using std::string;

#include "tensorflow/c/c_api.h"

#define NUM_INPUTS 1
#define NUM_OUTPUTS 1

class TFContext;

struct TestData {
	TFContext * ctx;

	int num_inputs;
	int num_outputs;
	TF_Output * input;
    TF_Output * output;
	TF_Tensor ** input_vals;
	TF_Tensor ** output_vals;

	int ndims;
	int64_t * dims;

	int ndata;
	float * data;

	TestData(TFContext * ctx);
	~TestData();

	bool init_output(TF_Output ** p_outptr, int * p_num, int size, string op_name_prefix, bool mark_id = true);
	static void init_data(int64_t ** p_dims, int * p_ndims, float ** p_data, int * p_ndata);
	bool init_tensors();

	static void NoOpDeallocator(void* data, size_t a, void* b) { }
};

#endif /* tf/TestData.h */
