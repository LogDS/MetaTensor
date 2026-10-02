#!/usr/bin/env bash
pushd ../submodules/antlr
sudo apt install openjdk-17-jdk -y
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
export PATH=$JAVA_HOME/bin:$PATH
mvn clean install -DskipTests -Dcheckstyle.skip