module github.com/openfluke/welvet

go 1.22.5

require (
	github.com/eliben/go-sentencepiece v0.7.0
	github.com/openfluke/webgpu v1.0.4
)

require google.golang.org/protobuf v1.34.2

replace github.com/openfluke/webgpu => ../webgpu

replace github.com/eliben/go-sentencepiece => ./third_party/go-sentencepiece
