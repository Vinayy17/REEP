import { useEffect, useRef } from "react"
import { useLocation, useNavigate, useNavigationType } from "react-router-dom"
import { App } from "@capacitor/app"
import { Capacitor } from "@capacitor/core"
import { toast } from "sonner"

const DASHBOARD_ROUTE = "/"
const EXIT_CONFIRMATION_WINDOW_MS = 2000

export default function useAndroidBackButton() {
  const navigate = useNavigate()
  const location = useLocation()
  const navigationType = useNavigationType()
  const lastBackPressRef = useRef(0)

  useEffect(() => {
    if (Capacitor.getPlatform() !== "android") {
      return undefined
    }

    let active = true
    let listenerHandle

    const registerListener = async () => {
      listenerHandle = await App.addListener("backButton", ({ canGoBack }) => {
        if (!active) return

        const isDashboard = location.pathname === DASHBOARD_ROUTE

        if (!isDashboard) {
          if (canGoBack || window.history.length > 1 || navigationType !== "POP") {
            navigate(-1)
          } else {
            navigate(DASHBOARD_ROUTE, { replace: true })
          }
          return
        }

        const now = Date.now()
        if (now - lastBackPressRef.current < EXIT_CONFIRMATION_WINDOW_MS) {
          App.exitApp()
          return
        }

        lastBackPressRef.current = now
        toast("Press back again to exit")
      })
    }

    registerListener()

    return () => {
      active = false
      if (listenerHandle) {
        listenerHandle.remove()
      }
    }
  }, [location.pathname, navigate, navigationType])
}
