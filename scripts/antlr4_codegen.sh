#!/usr/bin/env bash
java -jar submodules/antlr4/tool/target/antlr4-4.13.2-complete.jar -Dlanguage=Cpp -visitor $1 #-o antlr4/cpp