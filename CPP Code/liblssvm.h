//
// MATLAB Compiler: 23.2 (R2023b)
// Date: Mon Oct 28 15:04:34 2024
// Arguments:
// "-B""macro_default""-W""cpplib:liblssvm""-T""link:lib""lssvm_fitness.m""predi
// ct_lssvm.m""preprocess_data.m""pso_optimize.m""train_lssvm.m""rbf_kernel.m"
//

#ifndef liblssvm_h
#define liblssvm_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_liblssvm_C_API 
#define LIB_liblssvm_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV liblssvmInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV liblssvmInitialize(void);
extern LIB_liblssvm_C_API 
void MW_CALL_CONV liblssvmTerminate(void);

extern LIB_liblssvm_C_API 
void MW_CALL_CONV liblssvmPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxLssvm_fitness(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxPredict_lssvm(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxPreprocess_data(int nlhs, mxArray *plhs[], int nrhs, mxArray 
                                     *prhs[]);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxPso_optimize(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxTrain_lssvm(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

extern LIB_liblssvm_C_API 
bool MW_CALL_CONV mlxRbf_kernel(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_liblssvm
#define PUBLIC_liblssvm_CPP_API __declspec(dllexport)
#else
#define PUBLIC_liblssvm_CPP_API __declspec(dllimport)
#endif

#define LIB_liblssvm_CPP_API PUBLIC_liblssvm_CPP_API

#else

#if !defined(LIB_liblssvm_CPP_API)
#if defined(LIB_liblssvm_C_API)
#define LIB_liblssvm_CPP_API LIB_liblssvm_C_API
#else
#define LIB_liblssvm_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_liblssvm_CPP_API void MW_CALL_CONV lssvm_fitness(int nargout, mwArray& score, const mwArray& trainX, const mwArray& trainY, const mwArray& C, const mwArray& gamma, const mwArray& taskType);

extern LIB_liblssvm_CPP_API void MW_CALL_CONV predict_lssvm(int nargout, mwArray& predicted, mwArray& scores, const mwArray& model, const mwArray& testX);

extern LIB_liblssvm_CPP_API void MW_CALL_CONV preprocess_data(int nargout, mwArray& processed_X_train, mwArray& processed_y_train, mwArray& processed_X_test, const mwArray& X_train, const mwArray& y_train, const mwArray& X_test);

extern LIB_liblssvm_CPP_API void MW_CALL_CONV pso_optimize(int nargout, mwArray& best_C, mwArray& best_gamma, const mwArray& X_train, const mwArray& y_train, const mwArray& num_particles, const mwArray& max_iters, const mwArray& C_min, const mwArray& C_max, const mwArray& gamma_min, const mwArray& gamma_max, const mwArray& taskType);

extern LIB_liblssvm_CPP_API void MW_CALL_CONV train_lssvm(int nargout, mwArray& model, const mwArray& trainX, const mwArray& trainY, const mwArray& C, const mwArray& gamma, const mwArray& taskType);

extern LIB_liblssvm_CPP_API void MW_CALL_CONV rbf_kernel(int nargout, mwArray& K, const mwArray& X1, const mwArray& X2, const mwArray& gamma);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
