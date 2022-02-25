#ifndef _TF_TFCONTEXT_H_
#define _TF_TFCONTEXT_H_

#include "tensorflow/c/c_api.h"

#include "TestData.h"

class TFContext {
	TF_Graph * graph;
	TF_Status * status;
	TF_Session* session;
	TF_SessionOptions * session_opts;
	TF_Buffer * run_opts;

public:
	TFContext();
	~TFContext();

	bool load_model(const char* saved_model_dir);
	void run_test(int round, bool save = false, const char * input_file = NULL, const char * output_file = NULL);

	friend class TestData;
};

#endif /* tf/TFContext.h */
