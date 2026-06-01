package com.anonymous.textremoverpoc

import com.facebook.react.TurboReactPackage
import com.facebook.react.bridge.NativeModule
import com.facebook.react.bridge.ReactApplicationContext
import com.facebook.react.module.model.ReactModuleInfo
import com.facebook.react.module.model.ReactModuleInfoProvider

class OpenCVProcessorPackage : TurboReactPackage() {

  override fun getModule(
    name: String,
    reactContext: ReactApplicationContext
  ): NativeModule? {
    return if (name == OpenCVProcessorModule.NAME) {
      OpenCVProcessorModule(reactContext)
    } else {
      null
    }
  }

  override fun getReactModuleInfoProvider(): ReactModuleInfoProvider {
    return ReactModuleInfoProvider {
      mapOf(
        OpenCVProcessorModule.NAME to ReactModuleInfo(
          OpenCVProcessorModule.NAME,
          OpenCVProcessorModule.NAME,
          false, // canOverrideExistingModule
          false, // needsEagerInit
          false, // hasConstants
          BuildConfig.IS_NEW_ARCHITECTURE_ENABLED
        )
      )
    }
  }
}