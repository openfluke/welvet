//go:build windows && amd64 && cgo

package nativecuda

/*
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct {int32_t code;char message[256];} cuda_status;
typedef struct {uint8_t* data;size_t size;} cuda_blob;
typedef struct {HMODULE lib;
 int32_t(*abi)(void);const char*(*version)(void);const char*(*identity)(void);
 cuda_status(*mount)(const uint8_t*,size_t,void**);
 cuda_status(*forward)(void*,const uint8_t*,size_t,cuda_blob*);
 cuda_status(*forward_pair)(void*,const uint8_t*,size_t,cuda_blob*);
 cuda_status(*backward)(void*,const uint8_t*,size_t,cuda_blob*);
 cuda_status(*credit)(void*,int32_t,double,const uint8_t*,size_t);
 cuda_status(*dense_proxy)(void*,int32_t,const uint8_t*,size_t,cuda_blob*);
 cuda_status(*apply_sgd)(void*,double,const uint8_t*,size_t);
 cuda_status(*train)(void*,int32_t,double,int32_t,const uint8_t*,size_t,double*);
 cuda_status(*export_weights)(void*,cuda_blob*);
 cuda_status(*sync)(void*,int32_t);
 void(*release)(void*);void(*free_blob)(cuda_blob*);
} cuda_lib;
static void cuda_load_err(char* error){
 DWORD e=GetLastError();FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,NULL,e,0,error,255,NULL);
 if(!error[0])snprintf(error,256,"LoadLibrary failed (%lu)",(unsigned long)e);
}
static cuda_lib* cuda_open(const char* path,char* error){
 cuda_lib* m=calloc(1,sizeof(*m));if(!m){strcpy(error,"out of memory");return NULL;}
 m->lib=LoadLibraryExA(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!m->lib){cuda_load_err(error);free(m);return NULL;}
 #define LOAD(field,name) do {FARPROC p=GetProcAddress(m->lib,name);if(!p){snprintf(error,256,"missing symbol: " name);FreeLibrary(m->lib);free(m);return NULL;}*(void**)(&m->field)=(void*)p;}while(0)
 LOAD(abi,"welvet_model_abi_version");LOAD(version,"welvet_version");LOAD(identity,"welvet_runtime_identity");LOAD(mount,"welvet_model_mount");LOAD(forward,"welvet_model_forward");LOAD(train,"welvet_model_train_mse");LOAD(export_weights,"welvet_model_export_weights");LOAD(sync,"welvet_model_sync");LOAD(release,"welvet_model_release");LOAD(free_blob,"welvet_cuda_blob_free");
 *(void**)(&m->forward_pair)=(void*)GetProcAddress(m->lib,"welvet_model_forward_pair");
 *(void**)(&m->backward)=(void*)GetProcAddress(m->lib,"welvet_model_backward");
 *(void**)(&m->credit)=(void*)GetProcAddress(m->lib,"welvet_model_train_credit");
 *(void**)(&m->dense_proxy)=(void*)GetProcAddress(m->lib,"welvet_model_dense_proxy");
 *(void**)(&m->apply_sgd)=(void*)GetProcAddress(m->lib,"welvet_model_apply_sgd");
 #undef LOAD
 if(m->abi()!=1){strcpy(error,"unsupported CUDA model ABI");FreeLibrary(m->lib);free(m);return NULL;}return m;
}
static void cuda_close(cuda_lib* m){if(m){if(m->lib)FreeLibrary(m->lib);free(m);}}
static cuda_status cuda_mount(cuda_lib*m,const uint8_t*p,size_t n,void**s){return m->mount(p,n,s);}
static cuda_status cuda_forward(cuda_lib*m,void*s,const uint8_t*p,size_t n,cuda_blob*b){return m->forward(s,p,n,b);}
static cuda_status cuda_segment(cuda_lib*m,void*s,int kind,double lr,const uint8_t*p,size_t n,cuda_blob*b){
 if(!m->forward_pair || !m->backward || !m->apply_sgd){cuda_status e={2,"CUDA library lacks segment routing extension; rebuild/install welvet.cuda"};return e;}
 if(kind>=10){if(m->credit)return m->credit(s,kind-10,lr,p,n);cuda_status e={2,"CUDA library lacks credit routing extension"};return e;}
 if(kind>=3){if(m->dense_proxy)return m->dense_proxy(s,kind-3,p,n,b);cuda_status e={2,"CUDA library lacks Dense proxy routing extension"};return e;}
 if(kind==0)return m->forward_pair(s,p,n,b);
 if(kind==1)return m->backward(s,p,n,b);
 return m->apply_sgd(s,lr,p,n);
}
static cuda_status cuda_train(cuda_lib*m,void*s,int mode,double lr,int ticks,const uint8_t*p,size_t n,double*loss){return m->train(s,mode,lr,ticks,p,n,loss);}
static cuda_status cuda_export(cuda_lib*m,void*s,cuda_blob*b){return m->export_weights(s,b);}
static cuda_status cuda_sync(cuda_lib*m,void*s,int event){return m->sync(s,event);}
static void cuda_release(cuda_lib*m,void*s){m->release(s);}
static void cuda_free(cuda_lib*m,cuda_blob*b){m->free_blob(b);}
static const char* cuda_version(cuda_lib*m){return m->version();}
static const char* cuda_identity(cuda_lib*m){return m->identity();}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type nativeDriver struct {
	lib     *C.cuda_lib
	session unsafe.Pointer
}

func openDriver(path string, payload []byte) (driver, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var message [256]C.char
	m := C.cuda_open(p, &message[0])
	if m == nil {
		return nil, fmt.Errorf("nativecuda: load: %s", C.GoString(&message[0]))
	}
	d := &nativeDriver{lib: m}
	if e := cudaError(C.cuda_mount(m, bytePtr(payload), C.size_t(len(payload)), &d.session)); e != nil {
		C.cuda_close(m)
		return nil, e
	}
	return d, nil
}
func bytePtr(p []byte) *C.uint8_t {
	if len(p) == 0 {
		return nil
	}
	return (*C.uint8_t)(unsafe.Pointer(&p[0]))
}
func cudaError(s C.cuda_status) error {
	if s.code == 0 {
		return nil
	}
	return &NativeError{Code: int(s.code), Message: C.GoString(&s.message[0])}
}
func (d *nativeDriver) forward(p []byte) ([]byte, error) {
	var b C.cuda_blob
	e := cudaError(C.cuda_forward(d.lib, d.session, bytePtr(p), C.size_t(len(p)), &b))
	if e != nil {
		return nil, e
	}
	defer C.cuda_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativecuda: oversized output")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) train(mode int, lr float64, ticks int, p []byte) (float64, error) {
	var loss C.double
	e := cudaError(C.cuda_train(d.lib, d.session, C.int(mode), C.double(lr), C.int(ticks), bytePtr(p), C.size_t(len(p)), &loss))
	return float64(loss), e
}
func (d *nativeDriver) export() ([]byte, error) {
	var b C.cuda_blob
	e := cudaError(C.cuda_export(d.lib, d.session, &b))
	if e != nil {
		return nil, e
	}
	defer C.cuda_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativecuda: oversized snapshot")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) sync(event int) error {
	return cudaError(C.cuda_sync(d.lib, d.session, C.int(event)))
}
func (d *nativeDriver) version() string  { return C.GoString(C.cuda_version(d.lib)) }
func (d *nativeDriver) identity() string { return C.GoString(C.cuda_identity(d.lib)) }
func (d *nativeDriver) close() {
	C.cuda_release(d.lib, d.session)
	C.cuda_close(d.lib)
	d.session = nil
	d.lib = nil
}

func (d *nativeDriver) segment(kind int, lr float64, p []byte) ([]byte, error) {
	var b C.cuda_blob
	err := cudaError(C.cuda_segment(d.lib, d.session, C.int(kind), C.double(lr), bytePtr(p), C.size_t(len(p)), &b))
	if err != nil {
		return nil, err
	}
	defer C.cuda_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativecuda: oversized segment result")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
