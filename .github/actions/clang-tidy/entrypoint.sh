#!/bin/sh

git submodule update --init --recursive

mkdir build && cd build

cmake -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang -DENABLE_CUDA=Off -DCMAKE_EXPORT_COMPILE_COMMANDS=On ..

/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-avoid-bind' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-deprecated-headers' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-loop-convert' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-make-shared' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-make-unique' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-pass-by-value' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-raw-string-literal' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-redundant-void-arg' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-replace-auto-ptr' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-shrink-to-fit' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-auto' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-bool-literals' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-default-member-init' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-emplace' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-equals-default' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-equals-delete' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-nullptr' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-override' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-transparent-functors' -fix
/run-clang-tidy.py -header-filter='.*' -checks='-*,modernize-use-using' -fix

cd ../


git config --global user.name "github-actions[bot]"
git config --global user.email "github-actions[bot]@users.noreply.github.com"
git add .
git commit -m "${GITHUB_ACTION}: lint fix"

git push origin ${GITHUB_REF}
