- rewrite internal docs
- store multilingual-ness, model license spdx (exposed via model:: consts)
- expose distance & similarity functions
- configurable parallelism threshold in ese-build, hardcoded to 128 for encoders
- move from path dependencies to crates dependencies (do this last, we'll publish the dependents of encoders first)
- helper functions to directly compare two asrefstr or one to an iterasrefstr

readme rewrite:

- this crate takes a while to compile, and adds a lot to binary size (though to use any of these models at all, you need to have the weights at runtime)
- we should explain the exported consts, functions, etc
- static embedding makes major tradeoffs in quality in the name of performance
  - we hope this crate being as fast as it is makes up for that
- model table needs better notes, or split up by family or something
  - NIFE compatible with teacher model, potion-code is a code model, some aren't multilingual, etc.
  - bekko-static-v1-xx models are ours (and recommended) and in the same fam as the bekko model
  - non-bekko-static-xx families should live in twisty-dropdowns just to keep it cleaner
- we may (and probably will) change the default model between major versions!
  - must specify a specific model feature to benefit from engine changes without breaking embedding compatibility
- licensing: crate encoders, ese etc are dual Apache/MIT, but every model has its own license you must respect, available as a const in the models module

later:

- multi-model support? no idea how to even reason about this...
- f16 feature (automatically gated on std's availability of the feature?)
