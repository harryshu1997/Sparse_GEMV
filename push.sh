# 推送可执行文件
adb push /home/monsterharry/Documents/sparse_gemv/build_android/inner_product_cpu /data/local/tmp/sparse_gemv/
adb push /home/monsterharry/Documents/sparse_gemv/build_android/inner_product_gpu /data/local/tmp/sparse_gemv/

# 可选：赋予执行权限
adb shell "chmod +x /data/local/tmp/sparse_gemv/inner_product_cpu"
adb shell "chmod +x /data/local/tmp/sparse_gemv/inner_product_gpu"

echo "Push finished."

adb shell "LD_LIBRARY_PATH=/data/local/tmp/sparse_gemv /data/local/tmp/sparse_gemv/inner_product_gpu"