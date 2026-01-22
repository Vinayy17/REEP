import { Outlet, Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "@/context/AuthContext";
import {
  LayoutDashboard,
  Package,
  Users,
  FileText,
  FolderTree,
  LogOut,
} from "lucide-react";
import { Button } from "@/components/ui/button";

const Layout = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();

  const navItems = [
    { path: "/", icon: LayoutDashboard, label: "Dashboard" },
    { path: "/products", icon: Package, label: "Products" },
    { path: "/customers", icon: Users, label: "Customers" },
    { path: "/invoices", icon: FileText, label: "Invoices" },
    { path: "/categories", icon: FolderTree, label: "Categories" },
    { path: "/inventory", icon: FolderTree, label: "Inventory" },
  ];

  const handleLogout = () => {
    logout();
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-background flex-col md:flex-row">
      {/* ================= MOBILE TOP NAV ================= */}
      <header className="md:hidden flex items-center justify-between px-4 py-3 border-b bg-paper">
        <h1 className="text-lg font-bold text-primary">Inven</h1>

        <nav className="flex gap-4">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

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
            );
          })}
        </nav>
      </header>

      {/* ================= DESKTOP SIDEBAR ================= */}
      <aside
        className="hidden md:flex group w-16 hover:w-64 transition-all duration-300
                   border-r border-border bg-paper flex-col overflow-hidden"
      >
        {/* Logo */}
        <div className="p-4 border-b border-border">
          <h1
            className="text-xl font-bold text-primary
                       opacity-0 group-hover:opacity-100
                       transition-opacity"
          >
            InvenTrack
          </h1>
          <p
            className="text-xs text-foreground-muted mt-1
                       opacity-0 group-hover:opacity-100
                       transition-opacity"
          >
            Inventory Management
          </p>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-2 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = location.pathname === item.path;

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
                <span
                  className="opacity-0 group-hover:opacity-100 transition-opacity"
                >
                  {item.label}
                </span>
              </Link>
            );
          })}
        </nav>

        {/* User + Logout */}
        <div className="p-3 border-t border-border">
          <div className="flex items-center gap-3 px-2 py-2">
            <div className="w-10 h-10 rounded-full bg-primary flex items-center justify-center text-primary-foreground font-bold">
              {user?.name?.charAt(0)?.toUpperCase()}
            </div>

            <div className="opacity-0 group-hover:opacity-100 transition-opacity">
              <p className="text-sm font-medium">{user?.name}</p>
              <p className="text-xs text-foreground-muted">{user?.email}</p>
            </div>
          </div>

          <Button
            onClick={handleLogout}
            variant="outline"
            className="w-full justify-center group-hover:justify-start gap-2 mt-2"
          >
            <LogOut className="w-4 h-4" />
            <span className="opacity-0 group-hover:opacity-100 transition-opacity">
              Logout
            </span>
          </Button>
        </div>
      </aside>

      {/* ================= MAIN CONTENT ================= */}
      <main className="flex-1 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
};

export default Layout;
