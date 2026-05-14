import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { useEffect, useState } from "react"
import { Toaster } from "@/components/ui/sonner"

import Login from "@/pages/Login"
import Dashboard from "@/pages/Dashboard"
import Products from "@/pages/Products"
import Customers from "@/pages/Customers"
import Employees from "@/pages/Employees"
import Invoices from "@/pages/Invoices"
import Categories from "@/pages/Categories"
import Inventory from "@/pages/Inventory"
import Requirements from "@/pages/Requirements"
import Accounts from "@/pages/Account"
import InvoiceCreateCompact from "@/components/InvoiceCreateCompact"

import Layout from "@/components/Layout"
import { AuthProvider, useAuth } from "@/context/AuthContext"
import useAndroidBackButton from "@/hooks/useAndroidBackButton"
import "@/App.css"


// ✅ Protected Route
const ProtectedRoute = ({ children }) => {
  const { user, loading } = useAuth()

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <p className="text-sm text-muted-foreground">Restoring session...</p>
      </div>
    )
  }

  return user ? children : <Navigate to="/login" replace />
}


// ✅ Responsive Hook (BEST WAY)
const useIsMobile = () => {
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768)

  useEffect(() => {
    const handleResize = () => setIsMobile(window.innerWidth < 768)
    window.addEventListener("resize", handleResize)
    return () => window.removeEventListener("resize", handleResize)
  }, [])

  return isMobile
}


// ✅ Wrapper for Invoice Route
const InvoiceRouteHandler = () => {
  const isMobile = useIsMobile()
  return isMobile ? <InvoiceCreateCompact /> : <Invoices />
}


function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <AppShell />
      </BrowserRouter>
    </AuthProvider>
  )
}

const AppShell = () => {
  useAndroidBackButton()

  return (
    <>
        <Routes>

          {/* PUBLIC */}
          <Route path="/login" element={<Login />} />

          {/* PROTECTED */}
          <Route
            path="/"
            element={
              <ProtectedRoute>
                <Layout />
              </ProtectedRoute>
            }
          >

            <Route index element={<Dashboard />} />
            <Route path="products" element={<Products />} />
            <Route path="customers" element={<Customers />} />
            <Route path="employees" element={<Employees />} />
            <Route path="requirements" element={<Requirements />} />
            <Route path="categories" element={<Categories />} />
            <Route path="inventory" element={<Inventory />} />

            {/* ✅ SMART INVOICE ROUTE */}
            <Route path="invoices" element={<InvoiceRouteHandler />} />

            {/* ✅ ACCOUNTS */}
            <Route path="accounts" element={<Accounts />} />

          </Route>

        </Routes>

        <Toaster position="top-left" />
    </>
  )
}

export default App
