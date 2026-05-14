import { Outlet, Link, useLocation, useNavigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { useAuth } from "@/context/AuthContext"

import {
  LayoutDashboard,
  Package,
  Users,
  User,
  Building2,
  FileText,
  FolderTree,
  ClipboardList,
  Wallet,
  LogOut,
  ChevronLeft,
  ChevronRight,
  Plus,
  RefreshCw,
  Download,
  Menu,
  X,Eye
} from "lucide-react"

import { Button } from "@/components/ui/button"
import { ToastProvider, ToastViewport } from "@/components/ui/toast"

const Layout = () => {

  const location = useLocation()
  const navigate = useNavigate()
  const { user, logout } = useAuth()
const currentTab = location.state?.tab || "create"
  const [collapsed, setCollapsed] = useState(false)
  const [mobileMenu, setMobileMenu] = useState(false)
  const searchParams = new URLSearchParams(location.search)
  const currentPeopleTab = searchParams.get("tab") === "suppliers" ? "suppliers" : "customers"
  const isPeopleRoute = location.pathname === "/customers" || location.pathname === "/employees"
  const [peopleOpen, setPeopleOpen] = useState(isPeopleRoute)

  useEffect(() => {
    if (isPeopleRoute) {
      setPeopleOpen(true)
    }
  }, [isPeopleRoute])

  const primaryNavItems = [
    { path: "/", icon: LayoutDashboard, label: "Dashboard" },
    { path: "/products", icon: Package, label: "Products" },
  ]

  const peopleItems = [
    { key: "customers", to: "/customers?tab=customers", icon: User, label: "Customers" },
    { key: "suppliers", to: "/customers?tab=suppliers", icon: Building2, label: "Suppliers" },
    { key: "employees", to: "/employees", icon: Users, label: "Employees" },
  ]

  const secondaryNavItems = [
    { path: "/requirements", icon: ClipboardList, label: "Requirements" },
    { path: "/invoices", icon: FileText, label: "Invoices" },
    { path: "/categories", icon: FolderTree, label: "Categories" },
    { path: "/inventory", icon: FolderTree, label: "Inventory" },
    { path: "/accounts", icon: Wallet, label: "Accounts" }
  ]

  const bottomNav = [
    { path: "/products", icon: Package, label: "Products" },
    { path: "/customers", to: "/customers?tab=customers", icon: Users, label: "People" },
    { path: "/inventory", icon: FolderTree, label: "Inventory" },
    { path: "/requirements", icon: ClipboardList, label: "Needs" },
  ]
const isMobile = typeof window !== "undefined" && window.innerWidth < 768
  const handleLogout = () => {
    logout()
    navigate("/login")
  }
const isInvoicePage =
  location.pathname === "/invoices" ||
  location.pathname === "/invoicecreateCompact"
const isProductsPage = location.pathname === "/products"
const isCustomersPage = location.pathname === "/customers"
const isEmployeesPage = location.pathname === "/employees"
const isCategoriesPage = location.pathname === "/categories"
  const handleFabClick = () => {
    if (location.pathname === "/customers") {
      const targetTab = currentPeopleTab === "suppliers" ? "suppliers" : "customers"
      navigate(`/customers?tab=${targetTab}&action=add`)
      return
    }

    if (location.pathname === "/employees") {
      navigate("/employees?action=add")
      return
    }

    if (location.pathname === "/products") {
      navigate("/products?action=add")
      return
    }

    if (location.pathname === "/categories") {
      navigate("/categories?action=add")
      return
    }

    navigate("/invoices", { state: { tab: "create" } })
  }
  const currentPeopleLabel =
    location.pathname === "/employees"
      ? "Employees"
      : currentPeopleTab === "suppliers"
        ? "Suppliers"
        : "Customers"
  const currentPage =
    isPeopleRoute
      ? currentPeopleLabel
      : [...primaryNavItems, ...secondaryNavItems].find((item) => item.path === location.pathname)?.label || "Outrans"

  const isPeopleItemActive = (key) => {
    if (key === "customers") {
      return location.pathname === "/customers" && currentPeopleTab !== "suppliers"
    }

    if (key === "suppliers") {
      return location.pathname === "/customers" && currentPeopleTab === "suppliers"
    }

    return location.pathname === "/employees"
  }

  const handlePeopleToggle = () => {
    if (collapsed) {
      setCollapsed(false)
      setPeopleOpen(true)
      return
    }

    setPeopleOpen((prev) => !prev)
  }

  return (
    <ToastProvider>
      <ToastViewport />

      <div className="flex min-h-screen bg-background flex-col md:flex-row">

        {/* MOBILE HEADER */}
<header
  className="md:hidden fixed top-0 left-0 right-0 
  pt-[env(safe-area-inset-top)] 
  backdrop-blur-xl bg-background/90 border-b z-50"
>
  {(() => {
    const isInvoicePage =
      location.pathname === "/invoices" ||
      location.pathname === "/invoicecreateCompact"
    const invoicePath = location.pathname === "/invoicecreateCompact" ? "/invoicecreateCompact" : "/invoices"
    const currentTab = location.state?.tab || "create"

    return (
      <div className="h-14 flex items-center px-3">

        {/* LEFT - MENU */}
        <button
          onClick={() => setMobileMenu(true)}
          className="p-2 rounded-lg hover:bg-muted transition"
        >
          <Menu className="w-6 h-6" />
        </button>

        {/* CENTER - TITLE */}
        <h1 className="flex-1 text-center text-lg font-bold text-primary">
          {currentPage}
        </h1>

        {/* RIGHT - ACTIONS (ONLY IN INVOICE PAGE) */}
        {isInvoicePage ? (
          <div className="flex items-center gap-2">
            <button
              onClick={() =>
                navigate(`${invoicePath}?action=export`, {
                  state: { ...(location.state || {}), tab: "list" },
                })
              }
              className="rounded-lg p-2 text-muted-foreground transition hover:bg-muted"
            >
              <Download className="w-5 h-5" />
            </button>

            {/* Draft */}
            <button
              onClick={() => navigate(invoicePath, { state: { tab: "drafts" } })}
              className={`p-2 rounded-lg transition ${
                currentTab === "drafts"
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground"
              }`}
            >
              <FileText className="w-5 h-5" />
            </button>

            {/* Create (MAIN BUTTON) */}
            <button
              onClick={() => navigate(invoicePath, { state: { tab: "create" } })}
              className={`w-10 h-10 flex items-center justify-center rounded-lg transition ${
                currentTab === "create"
                  ? "bg-primary text-white"
                  : "bg-primary/80 text-white"
              }`}
            >
              <Plus className="w-5 h-5" />
            </button>

            {/* View */}
            <button
              onClick={() => navigate(invoicePath, { state: { tab: "list" } })}
              className={`p-2 rounded-lg transition ${
                currentTab === "list"
                  ? "text-primary bg-primary/10"
                  : "text-muted-foreground"
              }`}
            >
              <Eye className="w-5 h-5" />
            </button>

          </div>
        ) : isProductsPage ? (
          <div className="flex items-center gap-2">
            <button
              onClick={() => navigate("/products?action=refresh")}
              className="rounded-lg p-2 text-muted-foreground transition hover:bg-muted"
            >
              <RefreshCw className="h-5 w-5" />
            </button>
            <button
              onClick={() => navigate("/products?action=combo")}
              className="rounded-lg p-2 text-muted-foreground transition hover:bg-muted"
            >
              <Package className="h-5 w-5" />
            </button>
            <button
              onClick={() => navigate("/products?action=add")}
              className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-white"
            >
              <Plus className="h-5 w-5" />
            </button>
          </div>
        ) : isCustomersPage ? (
          <div className="w-10" />
        ) : isEmployeesPage ? (
          <div className="w-10" />
        ) : isCategoriesPage ? (
          <button
            onClick={() => navigate("/categories?action=add")}
            className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-white"
          >
            <Plus className="h-5 w-5" />
          </button>
        ) : (
          <button
            onClick={() => navigate("/accounts")}
            className="text-xs bg-primary text-white px-3 py-2 rounded-lg"
          >
            Accounts
          </button>
        )}

      </div>
    )
  })()}
</header>

        {/* MOBILE DRAWER */}

        {mobileMenu && (
          <div className="fixed inset-0 z-50 flex">

            <div
              className="absolute inset-0 bg-black/40"
              onClick={() => setMobileMenu(false)}
            />

            <div className="relative w-72 bg-background h-full shadow-2xl p-5">

              <div className="flex justify-between items-center mb-6">
                <h2 className="font-semibold text-lg">Menu</h2>

                <button onClick={() => setMobileMenu(false)}>
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="flex items-center gap-3 mb-6">

                <div className="w-12 h-12 rounded-full bg-primary text-white flex items-center justify-center font-bold">
                  {user?.name?.charAt(0)?.toUpperCase()}
                </div>

                <div>
                  <p className="font-medium">{user?.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {user?.email}
                  </p>
                </div>

              </div>

              <nav className="space-y-2">

                {primaryNavItems.map((item) => {

                  const Icon = item.icon
                  const active = location.pathname === item.path

                  return (
                    <button
                      key={item.path}
                      onClick={() => {
                        navigate(item.path)
                        setMobileMenu(false)
                      }}
                      className={`flex items-center gap-3 w-full px-3 py-3 rounded-lg transition ${
                        active
                          ? "bg-primary text-white"
                          : "hover:bg-muted"
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      {item.label}
                    </button>
                  )
                })}

                <div className="rounded-xl border border-white/5 bg-white/[0.02]">
                  <button
                    onClick={() => setPeopleOpen((prev) => !prev)}
                    className={`flex w-full items-center gap-3 px-3 py-3 rounded-xl transition ${
                      isPeopleRoute
                        ? "bg-primary text-white"
                        : "text-muted-foreground hover:bg-muted"
                    }`}
                  >
                    <Users className="w-5 h-5" />
                    <span className="flex-1 text-left">People</span>
                    <ChevronRight
                      className={`h-4 w-4 transition-transform ${peopleOpen ? "rotate-90" : ""}`}
                    />
                  </button>

                  {peopleOpen && (
                    <div className="space-y-1 px-2 pb-2">
                      {peopleItems.map((item) => {
                        const Icon = item.icon
                        const active = isPeopleItemActive(item.key)

                        return (
                          <button
                            key={item.key}
                            onClick={() => {
                              navigate(item.to)
                              setMobileMenu(false)
                            }}
                            className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm transition ${
                              active
                                ? "bg-primary/15 text-primary"
                                : "text-muted-foreground hover:bg-muted"
                            }`}
                          >
                            <Icon className="h-4 w-4" />
                            {item.label}
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>

                {secondaryNavItems.map((item) => {

                  const Icon = item.icon
                  const active = location.pathname === item.path

                  return (
                    <button
                      key={item.path}
                      onClick={() => {
                        navigate(item.path)
                        setMobileMenu(false)
                      }}
                      className={`flex items-center gap-3 w-full px-3 py-3 rounded-lg transition ${
                        active
                          ? "bg-primary text-white"
                          : "hover:bg-muted"
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                      {item.label}
                    </button>
                  )
                })}

              </nav>

              <Button
                onClick={handleLogout}
                variant="outline"
                className="w-full mt-8"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>

            </div>

          </div>
        )}

        {/* DESKTOP SIDEBAR */}

        <aside
          className={`hidden md:flex ${
            collapsed ? "w-16" : "w-64"
          } transition-all duration-300 border-r bg-background flex-col`}
        >

          <div className="p-4 border-b">
            <h1 className="text-xl font-bold text-primary">
              {collapsed ? "O" : "Outrans"}
            </h1>
          </div>

          <nav className="flex-1 p-2 space-y-1">

            {primaryNavItems.map((item) => {

              const Icon = item.icon
              const active = location.pathname === item.path

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-3 rounded-md ${
                    active
                      ? "bg-primary text-white"
                      : "hover:bg-muted text-muted-foreground"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {!collapsed && item.label}
                </Link>
              )
            })}

            <div className="space-y-1">
              <button
                onClick={handlePeopleToggle}
                className={`flex w-full items-center gap-3 rounded-md px-3 py-3 transition ${
                  isPeopleRoute
                    ? "bg-primary text-white"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                <Users className="w-5 h-5 shrink-0" />
                {!collapsed && (
                  <>
                    <span className="flex-1 text-left">People</span>
                    <ChevronRight
                      className={`h-4 w-4 transition-transform ${peopleOpen ? "rotate-90" : ""}`}
                    />
                  </>
                )}
              </button>

              {!collapsed && peopleOpen && (
                <div className="space-y-1 pl-4">
                  {peopleItems.map((item) => {
                    const Icon = item.icon
                    const active = isPeopleItemActive(item.key)

                    return (
                      <Link
                        key={item.key}
                        to={item.to}
                        className={`flex items-center gap-3 rounded-md px-3 py-2.5 text-sm transition ${
                          active
                            ? "bg-primary/15 text-primary"
                            : "text-muted-foreground hover:bg-muted"
                        }`}
                      >
                        <Icon className="h-4 w-4" />
                        {item.label}
                      </Link>
                    )
                  })}
                </div>
              )}
            </div>

            {secondaryNavItems.map((item) => {

              const Icon = item.icon
              const active = location.pathname === item.path

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`flex items-center gap-3 px-3 py-3 rounded-md ${
                    active
                      ? "bg-primary text-white"
                      : "hover:bg-muted text-muted-foreground"
                  }`}
                >
                  <Icon className="w-5 h-5" />
                  {!collapsed && item.label}
                </Link>
              )
            })}

          </nav>

          <Button
            onClick={() => setCollapsed(!collapsed)}
            variant="ghost"
            size="icon"
            className="mx-auto mb-4"
          >
            {collapsed ? <ChevronRight /> : <ChevronLeft />}
          </Button>

        </aside>

        {/* MAIN CONTENT */}

        <main className="flex-1 overflow-auto pt-20 md:pt-0 pb-32 px-4 md:px-6 md:pb-8">
          <Outlet />
        </main>

        {/* MOBILE BOTTOM NAV */}

      {!(
  isInvoicePage &&
  typeof window !== "undefined" &&
  window.innerWidth < 768
) && (
  <nav className="md:hidden fixed bottom-0 left-0 right-0 h-20 border-t bg-background/92 px-6 backdrop-blur-xl supports-[backdrop-filter]:bg-background/80 flex items-center justify-between z-50">

    {/* LEFT ITEMS */}
    {bottomNav.slice(0, 2).map((item) => {
      const Icon = item.icon
      const active = location.pathname === item.path

      return (
        <Link
          key={item.path}
          to={item.to || item.path}
          className={`flex flex-col items-center text-xs transition ${
            active ? "text-primary scale-110" : "text-muted-foreground"
          }`}
        >
          <Icon className="w-5 h-5 mb-1" />
          {item.label}
        </Link>
      )
    })}

    {/* CENTER BUTTON */}
    <button
 onClick={handleFabClick}
   className="absolute left-1/2 -translate-x-1/2 -top-6 bg-primary text-white w-14 h-14 rounded-full flex items-center justify-center shadow-lg hover:scale-110 transition"

    >
      <Plus className="w-6 h-6" />
    </button>

    {/* RIGHT ITEMS */}
    {bottomNav.slice(2).map((item) => {
      const Icon = item.icon
      const active = location.pathname === item.path

      return (
        <Link
          key={item.path}
          to={item.path}
          className={`flex flex-col items-center text-xs transition ${
            active ? "text-primary scale-110" : "text-muted-foreground"
          }`}
        >
          <Icon className="w-5 h-5 mb-1" />
          {item.label}
        </Link>
      )
    })}

  </nav>
)}

      </div>
    </ToastProvider>
  )
}

export default Layout
