//go:build windows && amd64 && cgo

package nativerocm

/*
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
typedef struct {int32_t code;char message[256];} rocm_status;
typedef struct {uint8_t* data;size_t size;} rocm_blob;
typedef struct {HMODULE lib;
 int32_t(*abi)(void);const char*(*version)(void);const char*(*identity)(void);
 rocm_status(*mount)(const uint8_t*,size_t,void**);
 rocm_status(*forward)(void*,const uint8_t*,size_t,rocm_blob*);
 rocm_status(*forward_pair)(void*,const uint8_t*,size_t,rocm_blob*);
 rocm_status(*backward)(void*,const uint8_t*,size_t,rocm_blob*);
 rocm_status(*credit)(void*,int32_t,double,const uint8_t*,size_t);
 rocm_status(*dense_proxy)(void*,int32_t,const uint8_t*,size_t,rocm_blob*);
 rocm_status(*apply_sgd)(void*,double,const uint8_t*,size_t);
 rocm_status(*train)(void*,int32_t,double,int32_t,const uint8_t*,size_t,double*);
 rocm_status(*export_weights)(void*,rocm_blob*);
 rocm_status(*sync)(void*,int32_t);
 void(*release)(void*);void(*free_blob)(rocm_blob*);
} rocm_lib;
static void rocm_load_err(char* error){
 DWORD e=GetLastError();FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,NULL,e,0,error,255,NULL);
 if(!error[0])snprintf(error,256,"LoadLibrary failed (%lu)",(unsigned long)e);
}
static rocm_lib* rocm_open(const char* path,char* error){
 rocm_lib* m=calloc(1,sizeof(*m));if(!m){strcpy(error,"out of memory");return NULL;}
 m->lib=LoadLibraryExA(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
 if(!m->lib){rocm_load_err(error);free(m);return NULL;}
 #define LOAD(field,name) do {FARPROC p=GetProcAddress(m->lib,name);if(!p){snprintf(error,256,"missing symbol: " name);FreeLibrary(m->lib);free(m);return NULL;}*(void**)(&m->field)=(void*)p;}while(0)
 LOAD(abi,"welvet_model_abi_version");LOAD(version,"welvet_version");LOAD(identity,"welvet_runtime_identity");LOAD(mount,"welvet_model_mount");LOAD(forward,"welvet_model_forward");LOAD(train,"welvet_model_train_mse");LOAD(export_weights,"welvet_model_export_weights");LOAD(sync,"welvet_model_sync");LOAD(release,"welvet_model_release");LOAD(free_blob,"welvet_rocm_blob_free");
 *(void**)(&m->forward_pair)=(void*)GetProcAddress(m->lib,"welvet_model_forward_pair");
 *(void**)(&m->backward)=(void*)GetProcAddress(m->lib,"welvet_model_backward");
 *(void**)(&m->credit)=(void*)GetProcAddress(m->lib,"welvet_model_train_credit");
 *(void**)(&m->dense_proxy)=(void*)GetProcAddress(m->lib,"welvet_model_dense_proxy");
 *(void**)(&m->apply_sgd)=(void*)GetProcAddress(m->lib,"welvet_model_apply_sgd");
 #undef LOAD
 if(m->abi()!=1){strcpy(error,"unsupported ROCm model ABI");FreeLibrary(m->lib);free(m);return NULL;}return m;
}
static void rocm_close(rocm_lib* m){if(m){if(m->lib)FreeLibrary(m->lib);free(m);}}
static rocm_status rocm_mount(rocm_lib*m,const uint8_t*p,size_t n,void**s){return m->mount(p,n,s);}
static rocm_status rocm_forward(rocm_lib*m,void*s,const uint8_t*p,size_t n,rocm_blob*b){return m->forward(s,p,n,b);}
static rocm_status rocm_segment(rocm_lib*m,void*s,int kind,double lr,const uint8_t*p,size_t n,rocm_blob*b){
 if(!m->forward_pair || !m->backward || !m->apply_sgd){rocm_status e={2,"ROCm library lacks segment routing extension; rebuild/install welvet.rocm"};return e;}
 if(kind>=10){if(m->credit)return m->credit(s,kind-10,lr,p,n);rocm_status e={2,"ROCm library lacks credit routing extension"};return e;}
 if(kind>=3){if(m->dense_proxy)return m->dense_proxy(s,kind-3,p,n,b);rocm_status e={2,"ROCm library lacks Dense proxy routing extension"};return e;}
 if(kind==0)return m->forward_pair(s,p,n,b);
 if(kind==1)return m->backward(s,p,n,b);
 return m->apply_sgd(s,lr,p,n);
}
static rocm_status rocm_train(rocm_lib*m,void*s,int mode,double lr,int ticks,const uint8_t*p,size_t n,double*loss){return m->train(s,mode,lr,ticks,p,n,loss);}
static rocm_status rocm_export(rocm_lib*m,void*s,rocm_blob*b){return m->export_weights(s,b);}
static rocm_status rocm_sync(rocm_lib*m,void*s,int event){return m->sync(s,event);}
static void rocm_release(rocm_lib*m,void*s){m->release(s);}
static void rocm_free(rocm_lib*m,rocm_blob*b){m->free_blob(b);}
static const char* rocm_version(rocm_lib*m){return m->version();}
static const char* rocm_identity(rocm_lib*m){return m->identity();}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type nativeDriver struct {
	lib     *C.rocm_lib
	session unsafe.Pointer
}

func openDriver(path string, payload []byte) (driver, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var message [256]C.char
	m := C.rocm_open(p, &message[0])
	if m == nil {
		return nil, fmt.Errorf("nativerocm: load: %s", C.GoString(&message[0]))
	}
	d := &nativeDriver{lib: m}
	if e := rocmError(C.rocm_mount(m, bytePtr(payload), C.size_t(len(payload)), &d.session)); e != nil {
		C.rocm_close(m)
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
func rocmError(s C.rocm_status) error {
	if s.code == 0 {
		return nil
	}
	return &NativeError{Code: int(s.code), Message: C.GoString(&s.message[0])}
}
func (d *nativeDriver) forward(p []byte) ([]byte, error) {
	var b C.rocm_blob
	e := rocmError(C.rocm_forward(d.lib, d.session, bytePtr(p), C.size_t(len(p)), &b))
	if e != nil {
		return nil, e
	}
	defer C.rocm_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativerocm: oversized output")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) train(mode int, lr float64, ticks int, p []byte) (float64, error) {
	var loss C.double
	e := rocmError(C.rocm_train(d.lib, d.session, C.int(mode), C.double(lr), C.int(ticks), bytePtr(p), C.size_t(len(p)), &loss))
	return float64(loss), e
}
func (d *nativeDriver) export() ([]byte, error) {
	var b C.rocm_blob
	e := rocmError(C.rocm_export(d.lib, d.session, &b))
	if e != nil {
		return nil, e
	}
	defer C.rocm_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativerocm: oversized snapshot")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) sync(event int) error {
	return rocmError(C.rocm_sync(d.lib, d.session, C.int(event)))
}
func (d *nativeDriver) version() string  { return C.GoString(C.rocm_version(d.lib)) }
func (d *nativeDriver) identity() string { return C.GoString(C.rocm_identity(d.lib)) }
func (d *nativeDriver) close() {
	C.rocm_release(d.lib, d.session)
	C.rocm_close(d.lib)
	d.session = nil
	d.lib = nil
}

func (d *nativeDriver) segment(kind int, lr float64, p []byte) ([]byte, error) {
	var b C.rocm_blob
	err := rocmError(C.rocm_segment(d.lib, d.session, C.int(kind), C.double(lr), bytePtr(p), C.size_t(len(p)), &b))
	if err != nil {
		return nil, err
	}
	defer C.rocm_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativerocm: oversized segment result")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
