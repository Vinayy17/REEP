import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"

import Login from "@/pages/Login"
import Dashboard from "@/pages/Dashboard"
import Products from "@/pages/Products"
import Customers from "@/pages/Customers"
import Invoices from "@/pages/Invoices"
import Categories from "@/pages/Categories"
import Inventory from "@/pages/Inventory"
import Requirements from "@/pages/Requirements"   // ✅ ADD

import Layout from "@/components/Layout"
import { AuthProvider, useAuth } from "@/context/AuthContext"
import "@/App.css"

const ProtectedRoute = ({ children }) => {
  const { user } = useAuth()
  return user ? children : <Navigate to="/login" />
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* ================= PUBLIC ================= */}
          <Route path="/login" element={<Login />} />

          {/* ================= PROTECTED ================= */}
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
            <Route path="requirements" element={<Requirements />} /> {/* ✅ NEW */}
            <Route path="invoices" element={<Invoices />} />
            <Route path="categories" element={<Categories />} />
            <Route path="inventory" element={<Inventory />} />
          </Route>
        </Routes>

        {/* Toasts */}
        <Toaster position="top-left" />
      </BrowserRouter>
    </AuthProvider>
  )
}

export default App
