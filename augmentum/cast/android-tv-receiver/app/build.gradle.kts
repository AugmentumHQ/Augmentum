/*
 * App module build — minimum viable Android TV target.
 *
 * compileSdk 34 (Android 14) covers every Onn box variant + every
 * modern Android TV / Google TV release. minSdk 24 (Nougat) is the
 * floor for WebView Auto-update; older boxes are vanishingly rare
 * for new installs.
 */
plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.augmentum.castreceiver"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.augmentum.castreceiver"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
        debug {
            // Debug builds are sideloaded directly via adb install —
            // we sign with the auto-generated debug keystore, no extra
            // signing config needed.
            isDebuggable = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    // AppCompat for AppCompatActivity. We deliberately skip Leanback
    // (too heavy for our needs; we're not building a real TV launcher
    // UI) and Compose (overkill for one config screen).
    implementation("androidx.appcompat:appcompat:1.6.1")
    // androidx.media for MediaSessionCompat + MediaButtonReceiver.
    // Registering a session lets the TV remote's transport keys
    // (play/pause/skip/ff/rew/stop) route into our PlaybackService
    // instead of being absorbed by whichever app last claimed the
    // audio focus. Also lights up Google Assistant + the system
    // now-playing card with whatever Augmentum is currently casting.
    implementation("androidx.media:media:1.7.0")
}
