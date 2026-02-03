import { Outlet, Link, useLocation, useNavigate } from "react-router-dom"
import { useState } from "react"
import { useAuth } from "@/context/AuthContext"
import {
  LayoutDashboard,
  Package,
  Users,
  FileText,
  FolderTree,
  ClipboardList,
  LogOut,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import { Button } from "@/components/ui/button"

// 🔥 Toast
import { ToastProvider, ToastViewport } from "@/components/ui/toast"

const Layout = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()

  const [collapsed, setCollapsed] = useState(false)

  const navItems = [
    { path: "/", icon: LayoutDashboard, label: "Dashboard" },
    { path: "/products", icon: Package, label: "Products" },
    { path: "/customers", icon: Users, label: "Customers" },
    { path: "/requirements", icon: ClipboardList, label: "Requirements" },
    { path: "/invoices", icon: FileText, label: "Invoices" },
    { path: "/categories", icon: FolderTree, label: "Categories" },
    { path: "/inventory", icon: FolderTree, label: "Inventory" },
  ]

  const handleLogout = () => {
    logout()
    navigate("/login")
  }

  return (
    <ToastProvider>
      <ToastViewport />

      <div className="flex h-screen bg-background flex-col md:flex-row">
        {/* ================= MOBILE TOP NAV ================= */}
        <header className="md:hidden flex items-center justify-between px-4 py-3 border-b bg-paper">
          <h1 className="text-lg font-bold text-primary">Outrans</h1>

          <nav className="flex gap-3">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`p-2 rounded-md ${
                    isActive
                      ? "bg-primary text-primary-foreground"
                      : "text-foreground-muted"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                </Link>
              )
            })}
          </nav>
        </header>

        {/* ================= DESKTOP SIDEBAR ================= */}
        <aside
          className={`hidden md:flex ${
            collapsed ? "w-16" : "w-64"
          } transition-all duration-300
          border-r border-border bg-paper flex-col`}
        >
          {/* Logo */}
          <div className="p-4 border-b border-border">
            <h1 className="text-xl font-bold text-primary">
              {collapsed ? "O" : "Outrans"}
            </h1>
            {!collapsed && (
              <p className="text-xs text-foreground-muted mt-1">
                Inventory Management
              </p>
            )}
          </div>

          {/* Navigation */}
          <nav className="flex-1 p-2 space-y-1">
            {navItems.map((item) => {
              const Icon = item.icon
              const isActive = location.pathname === item.path

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-3 rounded-md
                    ${
                      isActive
                        ? "bg-primary text-primary-foreground"
                        : "text-foreground-muted hover:bg-subtle hover:text-foreground"
                    }`}
                >
                  <Icon className="w-5 h-5 shrink-0" />
                  {!collapsed && <span>{item.label}</span>}
                </Link>
              )
            })}
          </nav>

          {/* User Section */}
          <div className="p-3 border-t border-border">
            <div className="flex items-center gap-3 px-2 py-2">
              <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
                {user?.name?.charAt(0)?.toUpperCase()}
              </div>

              {!collapsed && (
                <div>
                  <p className="text-sm font-medium">{user?.name}</p>
                  <p className="text-xs text-foreground-muted">{user?.email}</p>
                </div>
              )}
            </div>

            <Button
              onClick={handleLogout}
              variant="outline"
              className="w-full justify-center gap-2 mt-2"
            >
              <LogOut className="w-4 h-4" />
              {!collapsed && "Logout"}
            </Button>

            {/* Collapse Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setCollapsed(!collapsed)}
              className="mx-auto mt-3"
            >
              {collapsed ? (
                <ChevronRight className="w-4 h-4" />
              ) : (
                <ChevronLeft className="w-4 h-4" />
              )}
            </Button>
          </div>
        </aside>

        {/* ================= MAIN CONTENT ================= */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </ToastProvider>
  )
}

export default Layout
