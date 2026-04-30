"use client"

import { Capacitor } from "@capacitor/core"
import { Directory, Filesystem } from "@capacitor/filesystem"
import { LocalNotifications } from "@capacitor/local-notifications"
import { FileOpener } from "@capacitor-community/file-opener"

const DOWNLOADS_CHANNEL_ID = "downloads"
const OPEN_FILE_ACTION_TYPE = "OPEN_FILE"
const DEFAULT_DOWNLOAD_SUBPATH = "Download"

let notificationSetupPromise = null

const isNativePlatform = () => Capacitor.isNativePlatform()
const isAndroidNative = () => isNativePlatform() && Capacitor.getPlatform() === "android"

const blobToBase64 = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onloadend = () => {
      const result = reader.result
      if (typeof result !== "string") {
        reject(new Error("Failed to read file"))
        return
      }
      resolve(result.split(",")[1] || "")
    }
    reader.onerror = () => reject(new Error("Failed to convert file"))
    reader.readAsDataURL(blob)
  })

const triggerBrowserDownload = (blob, fileName) => {
  const downloadUrl = window.URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = downloadUrl
  anchor.download = fileName
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  window.URL.revokeObjectURL(downloadUrl)
}

const ensureFilesystemPermissions = async () => {
  if (!isNativePlatform() || typeof Filesystem?.checkPermissions !== "function") {
    return { publicStorage: "granted" }
  }

  const permission = await Filesystem.checkPermissions()
  if (permission.publicStorage === "granted") {
    return permission
  }

  const requested = await Filesystem.requestPermissions()
  if (requested.publicStorage !== "granted") {
    throw new Error("Storage permission denied")
  }

  return requested
}

const writeNativeFileWithFallbacks = async ({ path, data }) => {
  const attempts = [
    { directory: Directory.ExternalStorage, path },
    { directory: Directory.External, path },
    { directory: Directory.Documents, path },
  ]

  let lastError = null

  for (const attempt of attempts) {
    try {
      const result = await Filesystem.writeFile({
        path: attempt.path,
        data,
        directory: attempt.directory,
        recursive: true,
      })

      const uri = result.uri || (
        await Filesystem.getUri({
          directory: attempt.directory,
          path: attempt.path,
        })
      ).uri

      return {
        fileUri: uri,
        directory: attempt.directory,
        path: attempt.path,
      }
    } catch (error) {
      lastError = error
    }
  }

  throw lastError || new Error("Failed to save file")
}

const ensureNotificationPermissions = async () => {
  if (!isNativePlatform()) {
    return { display: "granted" }
  }

  const permission = await LocalNotifications.checkPermissions()
  if (permission.display === "granted") {
    return permission
  }

  return LocalNotifications.requestPermissions()
}

export const openSavedPdf = async (fileUri) => {
  if (!fileUri) {
    throw new Error("Saved PDF path not found")
  }

  if (!isNativePlatform()) {
    window.open(fileUri, "_blank", "noopener,noreferrer")
    return
  }

  await FileOpener.open({
    filePath: fileUri,
    contentType: "application/pdf",
    openWithDefault: true,
  })
}

export const initializePdfDownloadSupport = async () => {
  if (!isNativePlatform()) {
    return { notificationsGranted: true }
  }

  if (!notificationSetupPromise) {
    notificationSetupPromise = (async () => {
      const notificationPermission = await ensureNotificationPermissions()
      const notificationsGranted = notificationPermission.display === "granted"

      if (notificationsGranted) {
        await LocalNotifications.createChannel({
          id: DOWNLOADS_CHANNEL_ID,
          name: "Downloads",
          description: "Download notifications",
          importance: 5,
          visibility: 1,
        })

        await LocalNotifications.registerActionTypes({
          types: [
            {
              id: OPEN_FILE_ACTION_TYPE,
              actions: [{ id: "open", title: "Open" }],
            },
          ],
        })

        await LocalNotifications.addListener(
          "localNotificationActionPerformed",
          async (event) => {
            const fileUri = event.notification?.extra?.fileUri
            const mimeType = event.notification?.extra?.mimeType || "application/pdf"

            if (!fileUri) {
              return
            }

            try {
              if (mimeType === "application/pdf") {
                await openSavedPdf(fileUri)
                return
              }

              await FileOpener.open({
                filePath: fileUri,
                contentType: mimeType,
                openWithDefault: true,
              })
            } catch (error) {
              console.error("Failed to open downloaded file:", error)
            }
          }
        )
      }

      if (!notificationsGranted) {
        notificationSetupPromise = null
      }

      return { notificationsGranted }
    })().catch((error) => {
      notificationSetupPromise = null
      throw error
    })
  }

  return notificationSetupPromise
}

export const saveFileWithNotification = async ({
  blob,
  fileName,
  mimeType = "application/pdf",
  notificationTitle = "Invoice Downloaded 📄",
  notificationBody,
  relativePath,
}) => {
  if (!blob) {
    throw new Error("File data is missing")
  }

  if (!fileName) {
    throw new Error("File name is required")
  }

  if (!isNativePlatform()) {
    triggerBrowserDownload(blob, fileName)
    return {
      fileUri: null,
      notificationsGranted: true,
      notified: false,
    }
  }

  await ensureFilesystemPermissions()

  const base64Data = await blobToBase64(blob)
  const normalizedFileName = String(fileName).split(/[\\/]/).pop() || fileName
  const normalizedRelativePath = (relativePath || `${DEFAULT_DOWNLOAD_SUBPATH}/${normalizedFileName}`)
    .replace(/^\/+/, "")
    .replace(/\\/g, "/")
  const path = normalizedRelativePath.startsWith(`${DEFAULT_DOWNLOAD_SUBPATH}/`)
    ? normalizedRelativePath
    : `${DEFAULT_DOWNLOAD_SUBPATH}/${normalizedFileName}`
  const { fileUri } = await writeNativeFileWithFallbacks({
    path,
    data: base64Data,
  })

  if (isAndroidNative() && mimeType === "application/pdf") {
    try {
      await openSavedPdf(fileUri)
    } catch (error) {
      console.error("Failed to auto-open saved PDF:", error)
    }
  }

  let notificationsGranted = false
  let notified = false
  const resolvedNotificationBody = mimeType === "application/pdf"
    ? "PDF saved to Downloads"
    : (notificationBody || `${normalizedFileName} saved to Downloads`)

  try {
    const notificationState = await initializePdfDownloadSupport()
    notificationsGranted = Boolean(notificationState.notificationsGranted)

    if (notificationsGranted) {
      await LocalNotifications.schedule({
  notifications: [
    {
      id: Date.now(),
      title: notificationTitle,
      body: resolvedNotificationBody,
      channelId: DOWNLOADS_CHANNEL_ID,
      actionTypeId: OPEN_FILE_ACTION_TYPE,

      // 🔥 ADD THIS LINE (THIS IS THE FIX)
      schedule: { at: new Date(Date.now() + 1000) },

      extra: {
        fileUri,
        mimeType,
      },
    },
  ],
})
      notified = true
    }
  } catch (error) {
    console.error("Failed to schedule download notification:", error)
  }

  return {
    fileUri,
    notificationsGranted,
    notified,
  }
}

export const savePdfWithNotification = async ({ blob, fileName }) =>
  saveFileWithNotification({
    blob,
    fileName,
    mimeType: "application/pdf",
    notificationTitle: "Invoice Downloaded 📄",
    notificationBody: "PDF saved to Downloads",
  })
