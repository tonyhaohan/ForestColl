set -euo pipefail
runroot=$HOME/mcnf-runs/issue272_forestcoll_20261015
mkdir -p "$runroot/meta" "$runroot/tools"
cd "$runroot/tools"
curl -fL --connect-timeout 20 --max-time 300 -o jdk.tar.gz 'https://download.oracle.com/java/25/latest/jdk-25_linux-x64_bin.tar.gz' > "$runroot/meta/jdk-download.log" 2>&1
sha256sum jdk.tar.gz > "$runroot/meta/jdk.sha256"
tar -xzf jdk.tar.gz
jdkdir=$(find "$PWD" -maxdepth 1 -type d -name 'jdk-*' | head -1)
printf '%s\n' "$jdkdir" > "$runroot/meta/java-home.txt"
"$jdkdir/bin/java" -version > "$runroot/meta/java-version.txt" 2>&1
