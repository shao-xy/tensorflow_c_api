#include "tf/TFContext.h"

#define MODEL_DIR "../web-trace"
#define ROUND 50

int main(int argc, char * argv[])
{
	TFContext ctx;
	ctx.load_model(MODEL_DIR);
	//ctx.run_test(ROUND);
	ctx.run_test(ROUND, true, "data/input.txt", "data/output.txt");
	return 0;
}
