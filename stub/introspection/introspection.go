package introspection

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/openfluke/welvet/architecture"
)

// MethodInfo represents metadata about a method.
type MethodInfo struct {
	MethodName string          `json:"method_name"`
	Parameters []ParameterInfo `json:"parameters"`
	Returns    []string        `json:"returns"`
}

// ParameterInfo represents metadata about a parameter.
type ParameterInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// GetMethodsJSON returns JSON for all public methods on *architecture.Grid.
func GetMethodsJSON(g *architecture.Grid) (string, error) {
	methods, err := GetMethods(g)
	if err != nil {
		return "", fmt.Errorf("introspection: %w", err)
	}
	data, err := json.MarshalIndent(methods, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// GetMethods retrieves all public methods of *architecture.Grid.
func GetMethods(g *architecture.Grid) ([]MethodInfo, error) {
	if g == nil {
		return nil, fmt.Errorf("nil grid")
	}
	var methods []MethodInfo
	gridType := reflect.TypeOf(g)
	for i := 0; i < gridType.NumMethod(); i++ {
		method := gridType.Method(i)
		if method.Name[0] < 'A' || method.Name[0] > 'Z' {
			continue
		}
		var params []ParameterInfo
		methodType := method.Type
		for j := 1; j < methodType.NumIn(); j++ {
			params = append(params, ParameterInfo{
				Name: fmt.Sprintf("param%d", j-1),
				Type: methodType.In(j).String(),
			})
		}
		var returns []string
		for j := 0; j < methodType.NumOut(); j++ {
			returns = append(returns, methodType.Out(j).String())
		}
		methods = append(methods, MethodInfo{
			MethodName: method.Name,
			Parameters: params,
			Returns:    returns,
		})
	}
	return methods, nil
}

// GetMethodSignature returns the signature of a specific method.
func GetMethodSignature(g *architecture.Grid, methodName string) (string, error) {
	if g == nil {
		return "", fmt.Errorf("nil grid")
	}
	gridType := reflect.TypeOf(g)
	method, found := gridType.MethodByName(methodName)
	if !found {
		return "", fmt.Errorf("method %s not found", methodName)
	}
	methodType := method.Type
	params := []string{}
	for j := 1; j < methodType.NumIn(); j++ {
		params = append(params, methodType.In(j).String())
	}
	returns := []string{}
	for j := 0; j < methodType.NumOut(); j++ {
		returns = append(returns, methodType.Out(j).String())
	}
	signature := fmt.Sprintf("%s(%s)", methodName, joinStrings(params, ", "))
	if len(returns) > 0 {
		if len(returns) == 1 {
			signature += " " + returns[0]
		} else {
			signature += " (" + joinStrings(returns, ", ") + ")"
		}
	}
	return signature, nil
}

// ListMethods returns public method names on *architecture.Grid.
func ListMethods(g *architecture.Grid) []string {
	if g == nil {
		return nil
	}
	var names []string
	gridType := reflect.TypeOf(g)
	for i := 0; i < gridType.NumMethod(); i++ {
		method := gridType.Method(i)
		if method.Name[0] >= 'A' && method.Name[0] <= 'Z' {
			names = append(names, method.Name)
		}
	}
	return names
}

func joinStrings(strs []string, sep string) string {
	if len(strs) == 0 {
		return ""
	}
	result := strs[0]
	for i := 1; i < len(strs); i++ {
		result += sep + strs[i]
	}
	return result
}
