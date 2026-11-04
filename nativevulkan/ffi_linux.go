//go:build ((linux && amd64) || (android && (amd64 || arm64))) && cgo

package nativevulkan

/*
#include <dlfcn.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
// RTLD_DEEPBIND keeps welvet.vulkan C++ symbols from clashing with nativecpp libwelvet.so.
#ifndef RTLD_DEEPBIND
#define RTLD_DEEPBIND 0
#endif
typedef struct {int32_t code;char message[256];} vulkan_status;
typedef struct {uint8_t* data;size_t size;} vulkan_blob;
typedef struct {void* lib;
 vulkan_status(*init)(void);
 int32_t(*abi)(void);const char*(*version)(void);const char*(*identity)(void);
 vulkan_status(*mount)(const uint8_t*,size_t,void**);
 vulkan_status(*forward)(void*,const uint8_t*,size_t,vulkan_blob*);
 vulkan_status(*forward_pair)(void*,const uint8_t*,size_t,vulkan_blob*);
 vulkan_status(*backward)(void*,const uint8_t*,size_t,vulkan_blob*);
 vulkan_status(*credit)(void*,int32_t,double,const uint8_t*,size_t);
 vulkan_status(*dense_proxy)(void*,int32_t,const uint8_t*,size_t,vulkan_blob*);
 vulkan_status(*apply_sgd)(void*,double,const uint8_t*,size_t);
 vulkan_status(*train)(void*,int32_t,double,int32_t,const uint8_t*,size_t,double*);
 vulkan_status(*export_weights)(void*,vulkan_blob*);
 vulkan_status(*sync)(void*,int32_t);
 void(*release)(void*);void(*free_blob)(vulkan_blob*);
} vulkan_lib;
static vulkan_lib* vulkan_open(const char* path,char* error){
 vulkan_lib* m=calloc(1,sizeof(*m));if(!m){strcpy(error,"out of memory");return NULL;}
 m->lib=dlopen(path,RTLD_NOW|RTLD_LOCAL|RTLD_DEEPBIND);if(!m->lib){strncpy(error,dlerror(),255);free(m);return NULL;}
 #define LOAD(field,name) do {*(void**)(&m->field)=dlsym(m->lib,name);if(!m->field){strncpy(error,"missing symbol: " name,255);dlclose(m->lib);free(m);return NULL;}}while(0)
 LOAD(abi,"welvet_model_abi_version");LOAD(version,"welvet_version");LOAD(identity,"welvet_runtime_identity");LOAD(mount,"welvet_model_mount");LOAD(forward,"welvet_model_forward");LOAD(train,"welvet_model_train_mse");LOAD(export_weights,"welvet_model_export_weights");LOAD(sync,"welvet_model_sync");LOAD(release,"welvet_model_release");LOAD(free_blob,"welvet_vulkan_blob_free");
 *(void**)(&m->forward_pair)=dlsym(m->lib,"welvet_model_forward_pair");
 *(void**)(&m->backward)=dlsym(m->lib,"welvet_model_backward");
 *(void**)(&m->credit)=dlsym(m->lib,"welvet_model_train_credit");
 *(void**)(&m->dense_proxy)=dlsym(m->lib,"welvet_model_dense_proxy");
 *(void**)(&m->apply_sgd)=dlsym(m->lib,"welvet_model_apply_sgd");
 *(void**)(&m->init)=dlsym(m->lib,"welvet_vulkan_init");
 #undef LOAD
 if(m->abi()!=1){strcpy(error,"unsupported Vulkan model ABI");dlclose(m->lib);free(m);return NULL;}
 if(m->init){vulkan_status st=m->init();if(st.code!=0){strncpy(error,st.message,255);dlclose(m->lib);free(m);return NULL;}}
 return m;
}
static void vulkan_close(vulkan_lib* m){if(m){dlclose(m->lib);free(m);}}
static vulkan_status vulkan_mount(vulkan_lib*m,const uint8_t*p,size_t n,void**s){return m->mount(p,n,s);}
static vulkan_status vulkan_forward(vulkan_lib*m,void*s,const uint8_t*p,size_t n,vulkan_blob*b){return m->forward(s,p,n,b);}
static vulkan_status vulkan_segment(vulkan_lib*m,void*s,int kind,double lr,const uint8_t*p,size_t n,vulkan_blob*b){
 if(!m->forward_pair || !m->backward || !m->apply_sgd){vulkan_status e={2,"Vulkan library lacks segment routing extension; rebuild/install welvet.vulkan"};return e;}
 if(kind>=10){if(m->credit)return m->credit(s,kind-10,lr,p,n);vulkan_status e={2,"Vulkan library lacks credit routing extension"};return e;}
 if(kind>=3){if(m->dense_proxy)return m->dense_proxy(s,kind-3,p,n,b);vulkan_status e={2,"Vulkan library lacks Dense proxy routing extension"};return e;}
 if(kind==0)return m->forward_pair(s,p,n,b);
 if(kind==1)return m->backward(s,p,n,b);
 return m->apply_sgd(s,lr,p,n);
}
static vulkan_status vulkan_train(vulkan_lib*m,void*s,int mode,double lr,int ticks,const uint8_t*p,size_t n,double*loss){return m->train(s,mode,lr,ticks,p,n,loss);}
static vulkan_status vulkan_export(vulkan_lib*m,void*s,vulkan_blob*b){return m->export_weights(s,b);}
static vulkan_status vulkan_sync(vulkan_lib*m,void*s,int event){return m->sync(s,event);}
static void vulkan_release(vulkan_lib*m,void*s){m->release(s);}
static void vulkan_free(vulkan_lib*m,vulkan_blob*b){m->free_blob(b);}
static const char* vulkan_version(vulkan_lib*m){return m->version();}
static const char* vulkan_identity(vulkan_lib*m){return m->identity();}
*/
import "C"
import (
	"fmt"
	"unsafe"
)

type nativeDriver struct {
	lib     *C.vulkan_lib
	session unsafe.Pointer
}

func openDriver(path string, payload []byte) (driver, error) {
	p := C.CString(path)
	defer C.free(unsafe.Pointer(p))
	var message [256]C.char
	m := C.vulkan_open(p, &message[0])
	if m == nil {
		return nil, fmt.Errorf("nativevulkan: load: %s", C.GoString(&message[0]))
	}
	d := &nativeDriver{lib: m}
	if e := vulkanError(C.vulkan_mount(m, bytePtr(payload), C.size_t(len(payload)), &d.session)); e != nil {
		C.vulkan_close(m)
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
func vulkanError(s C.vulkan_status) error {
	if s.code == 0 {
		return nil
	}
	return &NativeError{Code: int(s.code), Message: C.GoString(&s.message[0])}
}
func (d *nativeDriver) forward(p []byte) ([]byte, error) {
	var b C.vulkan_blob
	e := vulkanError(C.vulkan_forward(d.lib, d.session, bytePtr(p), C.size_t(len(p)), &b))
	if e != nil {
		return nil, e
	}
	defer C.vulkan_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativevulkan: oversized output")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) train(mode int, lr float64, ticks int, p []byte) (float64, error) {
	var loss C.double
	e := vulkanError(C.vulkan_train(d.lib, d.session, C.int(mode), C.double(lr), C.int(ticks), bytePtr(p), C.size_t(len(p)), &loss))
	return float64(loss), e
}
func (d *nativeDriver) export() ([]byte, error) {
	var b C.vulkan_blob
	e := vulkanError(C.vulkan_export(d.lib, d.session, &b))
	if e != nil {
		return nil, e
	}
	defer C.vulkan_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativevulkan: oversized snapshot")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
func (d *nativeDriver) sync(event int) error {
	return vulkanError(C.vulkan_sync(d.lib, d.session, C.int(event)))
}
func (d *nativeDriver) version() string  { return C.GoString(C.vulkan_version(d.lib)) }
func (d *nativeDriver) identity() string { return C.GoString(C.vulkan_identity(d.lib)) }
func (d *nativeDriver) close() {
	C.vulkan_release(d.lib, d.session)
	C.vulkan_close(d.lib)
	d.session = nil
	d.lib = nil
}

func (d *nativeDriver) segment(kind int, lr float64, p []byte) ([]byte, error) {
	var b C.vulkan_blob
	err := vulkanError(C.vulkan_segment(d.lib, d.session, C.int(kind), C.double(lr), bytePtr(p), C.size_t(len(p)), &b))
	if err != nil {
		return nil, err
	}
	defer C.vulkan_free(d.lib, &b)
	if b.size > 1<<30 {
		return nil, fmt.Errorf("nativevulkan: oversized segment result")
	}
	return C.GoBytes(unsafe.Pointer(b.data), C.int(b.size)), nil
}
