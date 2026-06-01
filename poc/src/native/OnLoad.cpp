#include <jni.h>

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  return JNI_VERSION_1_6;
}
