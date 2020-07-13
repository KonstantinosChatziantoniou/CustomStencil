# CustomStencil


Create a gpu kernel for a stencil computation providing only the mathematical expression of the stencil.



## Usage

See `examples` folder. It contains the definition and usage of usual stencils.

There are 2 ways to define a stencil:

  + Give the mathematical expression
  + Give a coefficient matrix

### Mathematical expression

Give an expression to the macro `@def_stencil_expression`. For example
        
        stencil =  @def_stencil_expression c[0]*D[x,y,z] + c[1]*(D[x+1,y,z] + D[x-1,y,z])

You can alse use the defined `@sum` macro:
      
        stencil =   @def_stencil_expression c[0]*D[x,y,z] + @sum(i, 1, 2, c[i]*(D[x+i,y,z] + D[x-i,y,z])
Nested `@sum`s can also be used, but be careful for the Julia syntax. Additionally you can use `min max abs` for the coefficient indices (see `examples/ex_dense_stencil.jl`)

### Coefficients Array

You can provide a 3d coefficient array instead of expression. The coefficient array **must** have side length `2*radius + 1`.
So the center will be `(radius+1, radius+1, radius+1)`. See `examples/ex_star_coefs_stencil.jl`

### Applying the stencil

After you get the `StencilInstance` struct, by calling one of the methods of`NewStencilInstance`,
call the `ApplyStencil` and pass the struct and the padded data.

## Limitations
  + The data array name in the expression should always be `D` and the coef array `c`
  + Only works for x,y dimensions multiple of 16 (without padding)
  + User has to pad their own data according to the maximum radius.
  + User has to provide the maximum radius (although it can be extracted from the stencil expression) to be consistent with data padding.
  + Only 3d stencil is supported
  + For the mathematical expression, the coefficient Array, must be a `Vector` with length radius+1
  
## Possible Extensions
  
  + Create a higher order in space stencil combining time steps.
  + Add to kernel expressions
  
  
 ## How it works
  Given an `Expr` (Julia expression type), it modifies the expression to substitute the indices, and then evaluates it. The global scope `D` array
  contains a list of `SymEngine.symbols`. Then using symbolic math parsing, it gets the coefficient for each cell of the stencil.
  
  The code of the kernel is created by constructing and combining `quotes`. Then `SyntaxTree.genfun` is used to create a Function.
 
 
 
 ## Warning

Peek at the code of the module at your own risk!
