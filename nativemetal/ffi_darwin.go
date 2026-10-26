//go:build darwin && arm64 && cgo

package nativemetal

/*
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
typedef struct {int32_t code;char message[256];} metal_status;
typedef struct {uint8_t* data;size_t size;} metal_blob;
typedef struct {void* lib;
 int32_t(*abi)(void);const char*(*version)(void);const char*(*identity)(void);
 metal_status(*mount)(const uint8_t*,size_t,void**);
 metal_status(*forward)(void*,const uint8_t*,size_t,metal_blob*);
 metal_status(*forward_pair)(void*,const uint8_t*,size_t,metal_blob*);
 metal_status(*backward)(void*,const uint8_t*,size_t,metal_blob*);
 metal_status(*credit)(void*,int32_t,double,const uint8_t*,size_t);
 metal_status(*dense_proxy)(void*,int32_t,const uint8_t*,size_t,metal_blob*);
 metal_status(*apply_sgd)(void*,double,const uint8_t*,size_t);
 metal_status(*train)(void*,int32_t,double,int32_t,const uint8_t*,size_t,double*);
 metal_status(*export_weights)(void*,metal_blob*);
 metal_status(*sync)(void*,int32_t);
 void(*release)(void*);void(*free_blob)(metal_blob*);
} metal_lib;
static metal_lib* metal_open(const char* path,char* error){
 metal_lib* m=calloc(1,sizeof(*m));if(!m){strcpy(error,"out of memory");return NULL;}
 m->lib=dlopen(path,RTLD_NOW|RTLD_LOCAL);if(!m->lib){strncpy(error,dlerror(),255);free(m);return NULL;}
 #define LOAD(field,name) do {*(void**)(&m->field)=dlsym(m->lib,name);if(!m->field){strncpy(error,"missing symbol: " name,255);dlclose(m->lib);free(m);return NULL;}}while(0)
 LOAD(abi,"welvet_model_abi_version");LOAD(version,"welvet_version");LOAD(identity,"welvet_runtime_identity");LOAD(mount,"welvet_model_mount");LOAD(forward,"welvet_model_forward");LOAD(train,"welvet_model_train_mse");LOAD(export_weights,"welvet_model_export_weights");LOAD(sync,"welvet_model_sync");LOAD(release,"welvet_model_release");LOAD(free_blob,"welvet_metal_blob_free");
 *(void**)(&m->forward_pair)=dlsym(m->lib,"welvet_model_forward_pair");
 *(void**)(&m->backward)=dlsym(m->lib,"welvet_model_backward");
 *(void**)(&m->credit)=dlsym(m->lib,"welvet_model_train_credit");
 *(void**)(&m->dense_proxy)=dlsym(m->lib,"welvet_model_dense_proxy");
 *(void**)(&m->apply_sgd)=dlsym(m->lib,"welvet_model_apply_sgd");
 #undef LOAD
 if(m->abi()!=1){strcpy(error,"unsupported Metal model ABI");dlclose(m->lib);free(m);return NULL;}return m;
}
static void metal_close(metal_lib* m){if(m){dlclose(m->lib);free(m);}}
static metal_status metal_mount(metal_lib*m,const uint8_t*p,size_t n,void**s){return m->mount(p,n,s);}
static metal_status metal_forward(metal_lib*m,void*s,const uint8_t*p,size_t n,metal_blob*b){return m->forward(s,p,n,b);}
static metal_status metal_segment(metal_lib*m,void*s,int kind,double lr,const uint8_t*p,size_t n,metal_blob*b){
 if(!m->forward_pair || !m->backward || !m->apply_sgd){metal_status e={2,"Metal library lacks segment routing extension; rebuild/install welvet.metal"};return e;}
 if(kind>=10){if(m->credit)return m->credit(s,kind-10,lr,p,n);metal_status e={2,"Metal library lacks credit routing extension"};return e;}
 if(kind>=3){if(m->dense_proxy)return m->dense_proxy(s,kind-3,p,n,b);metal_status e={2,"Metal library lacks Dense proxy routing extension"};return e;}
 if(kind==0)return m->forward_pair(s,p,n,b);
 if(kind==1)return m->backward(s,p,n,b);
 return m->apply_sgd(s,lr,p,n);
}
static metal_status metal_train(metal_lib*m,void*s,int mode,double lr,int ticks,const uint8_t*p,size_t n,double*loss){return m->train(s,mode,lr,ticks,p,n,loss);}
static metal_status metal_export(metal_lib*m,void*s,metal_blob*b){return m->export_weights(s,b);}
static metal_status metal_sync(metal_lib*m,void*s,int event){return m->sync(s,event);}
static void metal_release(metal_lib*m,void*s){m->release(s);}
static void metal_free(metal_lib*m,metal_blob*b){m->free_blob(b);}
static const char* metal_version(metal_lib*m){return m->version();}
static const char* metal_identity(metal_lib*m){return m->identity();}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type nativeDriver struct {
	lib     *C.metal_lib
	session unsafe.Pointer
}

func openDriver(path string, payload []byte) (driver, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var message [256]C.char
	m := C.metal_open(p, &message[0])
	if m == nil {
		return nil, fmt.Errorf("nativemetal: load: %s", C.GoString(&message[0]))
	}
	d := &nativeDriver{lib: m}
	if e := metalError(C.metal_mount(m, bytePtr(payload), C.size_t(len(payload)), &d.session)); e != nil {
		C.metal_close(m)
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
func metalError(s C.metal_status) error {
	if s.code == 0 {
		return nil
	}
	return &NativeError{Code: int(s.code), Message: C.GoString(&s.message[0])}
}
func (d *nativeDriver) forward(p []byte) ([]byte, error) {
	var b C.metal_blob
	e := metalError(C.metal_forward(d.lib, d.session, bytePtr(p), C.size_t(len(p)), &b))
	if e != nil {
		return nil, e
	}
	defer C.metal_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativemetal: oversized output")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) train(mode int, lr float64, ticks int, p []byte) (float64, error) {
	var loss C.double
	e := metalError(C.metal_train(d.lib, d.session, C.int(mode), C.double(lr), C.int(ticks), bytePtr(p), C.size_t(len(p)), &loss))
	return float64(loss), e
}
func (d *nativeDriver) export() ([]byte, error) {
	var b C.metal_blob
	e := metalError(C.metal_export(d.lib, d.session, &b))
	if e != nil {
		return nil, e
	}
	defer C.metal_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativemetal: oversized snapshot")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) sync(event int) error {
	return metalError(C.metal_sync(d.lib, d.session, C.int(event)))
}
func (d *nativeDriver) version() string  { return C.GoString(C.metal_version(d.lib)) }
func (d *nativeDriver) identity() string { return C.GoString(C.metal_identity(d.lib)) }
func (d *nativeDriver) close() {
	C.metal_release(d.lib, d.session)
	C.metal_close(d.lib)
	d.session = nil
	d.lib = nil
}

func (d *nativeDriver) segment(kind int, lr float64, p []byte) ([]byte, error) {
	var b C.metal_blob
	err := metalError(C.metal_segment(d.lib, d.session, C.int(kind), C.double(lr), bytePtr(p), C.size_t(len(p)), &b))
	if err != nil {
		return nil, err
	}
	defer C.metal_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativemetal: oversized segment result")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
