"use client"

import { useEffect, useState, useRef } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "sonner"
import { Plus, Trash2, Search, Download, Eye, FileText, ScanLine, Loader2, Camera, X, Share2 } from "lucide-react"
import { Html5Qrcode, Html5QrcodeSupportedFormats } from "html5-qrcode"
import { useLocation } from "react-router-dom"

const API = `${process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"}/api`

const formatCurrency = (amount, decimals = 2) => {
  const num = typeof amount === "number" ? amount : Number(amount)
  return isNaN(num) ? "0.00" : num.toFixed(decimals)
}

const formatNumber = (amount, decimals = 0) => {
  const num = typeof amount === "number" ? amount : Number(amount)
  return isNaN(num) ? "0" : num.toFixed(decimals)
}

export default function Invoice() {
  const [activeTab, setActiveTab] = useState("create")
  const [invoices, setInvoices] = useState([])
  const [products, setProducts] = useState([])
  const [open, setOpen] = useState(false)
  const [viewInvoice, setViewInvoice] = useState(null)
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [page, setPage] = useState(1)
  const [statusFilter, setStatusFilter] = useState(null)
  const [rangeFilter, setRangeFilter] = useState(null)
  const [monthFilter, setMonthFilter] = useState(null)
  const [limit, setLimit] = useState(10)
  const [pagination, setPagination] = useState({ page: 1, total_pages: 1 })
  const [discount, setDiscount] = useState(0)
  const [paymentStatus, setPaymentStatus] = useState("pending")

  const [productSearch, setProductSearch] = useState("")
  const [searchResults, setSearchResults] = useState([])
  const [showProductSearch, setShowProductSearch] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [showVariantDialog, setShowVariantDialog] = useState(false)

  // Customer fields
  const [customerPhone, setCustomerPhone] = useState("")
  const [customerName, setCustomerName] = useState("")
  const [customerEmail, setCustomerEmail] = useState("")
  const [customerAddress, setCustomerAddress] = useState("")
  const [customerId, setCustomerId] = useState(null)

  // Invoice fields
  const [lineItems, setLineItems] = useState([])
  const [skuInput, setSkuInput] = useState("")
  const [isScanning, setIsScanning] = useState(false)
  const [showCamera, setShowCamera] = useState(false)
  const [cameraLoading, setCameraLoading] = useState(false)
  const [lastScannedSku, setLastScannedSku] = useState(null)
  const [lastScanTime, setLastScanTime] = useState(0)
  const scannerRef = useRef(null)
  const scanLockRef = useRef(false)
  const [scanMode, setScanMode] = useState(null) // "camera" | "barcode"
  const skuInputRef = useRef(null)

  const scanTimeoutRef = useRef(null)
  const location = useLocation()
  const [manualAmount, setManualAmount] = useState(0)
  const [manualLabel, setManualLabel] = useState("Additional Charge")

  const [productSearchQuery, setProductSearchQuery] = useState("")
  const [productSearchResults, setProductSearchResults] = useState([])
  const [isSearchingProducts, setIsSearchingProducts] = useState(false)
  const searchTimeoutRef = useRef(null)
  const [selectedProductForVariant, setSelectedProductForVariant] = useState(null)

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (token) {
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`
    }

    fetchProducts()
  }, [])

  useEffect(() => {
    fetchInvoices()
  }, [page, statusFilter, rangeFilter, monthFilter])

  useEffect(() => {
    if (activeTab === "create" && scanMode === "barcode") {
      setTimeout(() => skuInputRef.current?.focus(), 300)
    }
  }, [activeTab, scanMode])

  useEffect(() => {
    const isMobile = /Android|iPhone|iPad/i.test(navigator.userAgent)
    if (isMobile) {
      setScanMode("camera")
      setShowCamera(true)
    } else {
      setScanMode("barcode")
      setShowCamera(false)
      setTimeout(() => skuInputRef.current?.focus(), 300)
    }
  }, [])

  useEffect(() => {
    if (location.state?.customer) {
      const c = location.state.customer

      setCustomerId(c.id)
      setCustomerName(c.name || "")
      setCustomerPhone(c.phone || "")
      setCustomerEmail(c.email || "")
      setCustomerAddress(c.address || "")

      setActiveTab("create")

      toast.success("Customer details loaded")
    }
  }, [location.state])

  useEffect(() => {
    if (productSearchQuery.length >= 2) {
      if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current)
      searchTimeoutRef.current = setTimeout(() => {
        handleProductSearch(productSearchQuery)
      }, 300)
    } else {
      setProductSearchResults([])
    }
    return () => {
      if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current)
    }
  }, [productSearchQuery])

  const fetchInvoices = async () => {
    try {
      setLoading(true)

      const res = await axios.get(`${API}/invoices`, {
        params: {
          page,
          limit,
          status: statusFilter || undefined,
          range: rangeFilter || undefined,
          month: monthFilter || undefined,
        },
      })

      const parsedInvoices = (res.data?.data || []).map((inv) => {
        let items = []

        try {
          if (Array.isArray(inv.items)) {
            items = inv.items
          } else if (typeof inv.items === "string") {
            items = JSON.parse(inv.items)
          }
        } catch (e) {
          console.error("Item parse failed", e)
          items = []
        }

        return {
          ...inv,
          items,
          payment_status: inv.payment_status || "pending",
        }
      })

      setInvoices(parsedInvoices)
      setPagination(res.data.pagination)
    } catch (err) {
      console.error("Invoice fetch failed", err)
      toast.error(err?.response?.data?.detail || "Failed to load invoices")
    } finally {
      setLoading(false)
    }
  }

  const fetchProducts = async () => {
    try {
      const res = await axios.get(`${API}/products`)
      setProducts(res.data)
    } catch (err) {
      console.error("Product fetch failed", err)
      toast.error("Failed to load products")
    }
  }

  const handleProductSearch = async (query) => {
    try {
      setIsSearchingProducts(true)
      const res = await axios.get(`${API}/products/search?q=${query}`)
      setProductSearchResults(res.data || [])
    } catch (err) {
      console.error("Product search failed:", err)
    } finally {
      setIsSearchingProducts(false)
    }
  }

  const selectProduct = (product) => {
    if (product.variants && product.variants.length > 0) {
      setSelectedProduct(product)
      setShowVariantDialog(true)
    } else {
      addProductToInvoice(product)
    }
    setProductSearch("")
    setSearchResults([])
  }

  const addProductToInvoice = (product, variant = null) => {
    const newItem = {
      product_id: product.id,
      sku: variant ? variant.v_sku : product.sku,
      product_name: variant
        ? `${product.name} (${variant.variant_name || variant.color || variant.size || "Variant"})`
        : product.name,
      quantity: 1,
      price: variant?.v_selling_price || product.selling_price,
      gst_rate: 18,
      total: variant?.v_selling_price || product.selling_price,
      is_service: 0,
      image_url: variant?.image_url || (product.images && product.images[0]) || "/placeholder.png",
      v_sku: variant ? variant.v_sku : null,
      variant_name: variant ? variant.variant_name : null,
      color: variant ? variant.color : null,
      size: variant ? variant.size : null,
      variant_info: variant,
    }

    setLineItems((prev) => {
      const existingIdx = prev.findIndex((item) => (item.v_sku || item.sku) === newItem.sku)
      if (existingIdx !== -1) {
        const updated = [...prev]
        updated[existingIdx].quantity += 1
        updated[existingIdx].total = updated[existingIdx].quantity * updated[existingIdx].price
        return updated
      }
      return [...prev, newItem]
    })

    toast.success(`${newItem.product_name} added`)
    setShowVariantDialog(false)
    setSelectedProduct(null)
  }

  const searchCustomerByPhone = async (phone) => {
    if (phone.length < 10) return

    try {
      const res = await axios.get(`${API}/customers/search?phone=${phone}`)

      if (res.data) {
        setCustomerId(res.data.id)
        setCustomerName(res.data.name)
        setCustomerEmail(res.data.email)
        setCustomerAddress(res.data.address || "")
        toast.success("Customer found!")
      } else {
        setCustomerId(null)
        setCustomerName("")
        setCustomerEmail("")
        setCustomerAddress("")
        toast.info("New customer - please fill details")
      }
    } catch (err) {
      console.error("Customer search failed", err)
      toast.error("Customer lookup failed")
    }
  }

  const addLineItem = () => {
    setLineItems((prev) => [
      ...prev,
      {
        product_id: "",
        product_name: "",
        quantity: 1,
        price: 0,
        gst_rate: 18,
        total: 0,
        is_service: 0,
        v_sku: null,
        variant_name: null,
        color: null,
        size: null,
        variant_info: null, // Added for compatibility with updates
      },
    ])
  }

  const updateLineItem = (index, field, value) => {
    const updated = [...lineItems]
    updated[index][field] = value

    if (field === "product_id") {
      const product = products.find((p) => String(p.id) === String(value))
      if (product) {
        const hasVariants = product.variants && product.variants.length > 0

        if (hasVariants && product.variants.length === 1) {
          // If only one variant, auto-select it
          const variant = product.variants[0]
          updated[index].product_name =
            `${product.name} (${variant.variant_name || variant.color || variant.size || "Variant"})`
          updated[index].price = variant.v_selling_price || product.selling_price
          updated[index].v_sku = variant.v_sku
          updated[index].variant_name = variant.variant_name
          updated[index].color = variant.color
          updated[index].size = variant.size
          updated[index].image_url = variant.image_url || (product.images && product.images[0]) || "/placeholder.png"
          updated[index].variant_info = variant // Store variant info for compatibility
        } else {
          // Regular product or multiple variants (show base product)
          updated[index].product_name = product.name
          updated[index].price = product.selling_price
          updated[index].v_sku = null
          updated[index].variant_name = null
          updated[index].color = null
          updated[index].size = null
          updated[index].image_url = (product.images && product.images[0]) || "/placeholder.png"
          updated[index].variant_info = null // Clear variant info
        }

        updated[index].total = updated[index].price * updated[index].quantity
      }
    }

    if (field === "quantity" || field === "price") {
      updated[index].total = updated[index].price * updated[index].quantity
    }

    setLineItems(updated)
  }

 const handleSkuScan = async (skuInputRaw) => {
  if (!skuInputRaw) return
  if (scanLockRef.current) return
  scanLockRef.current = true

  let scannedSku = skuInputRaw.trim()

  // ✅ Handle QR JSON payload
  try {
    const parsed = JSON.parse(scannedSku)
    scannedSku = parsed.v_sku || parsed.sku || scannedSku
  } catch {}

  try {
    const res = await fetch(`${API}/products/sku/${encodeURIComponent(scannedSku)}`, {
      headers: {
        Authorization: `Bearer ${localStorage.getItem("token")}`,
      },
    })

    if (!res.ok) throw new Error("Not found")
    const product = await res.json()

    const isVariant = !!product.variant
    const hasVariants = Array.isArray(product.variants) && product.variants.length > 0

    // 🚨 CASE 1: PRODUCT QR / PRODUCT SKU WITH VARIANTS
    if (!isVariant && hasVariants) {
      setSelectedProductForVariant(product)
      setShowVariantDialog(true)
      setSkuInput("")
      return
    }

    // 🟢 CASE 2: VARIANT QR / VARIANT SKU
    let displayName, itemPrice, itemImage, variantInfo

    if (isVariant) {
      const variant = product.variant
      const variantLabel = variant.variant_name || variant.color || variant.size || "Variant"

      displayName = `${product.name} (${variantLabel})`
      itemPrice = variant.v_selling_price || product.selling_price
      itemImage =
        variant.image_url ||
        variant.v_image_url ||
        product.images?.[0] ||
        "/placeholder.png"

      variantInfo = {
        v_sku: variant.v_sku,
        variant_name: variant.variant_name,
        color: variant.color,
        size: variant.size,
        v_selling_price: variant.v_selling_price,
      }
    } else {
      // 🟢 CASE 3: PRODUCT WITHOUT VARIANTS
      displayName = product.name
      itemPrice = product.selling_price
      itemImage = product.images?.[0] || "/placeholder.png"
      variantInfo = null
    }

    setLineItems((prev) => {
      const key = isVariant ? variantInfo.v_sku : product.sku
      const idx = prev.findIndex((i) => (i.v_sku || i.sku) === key)

      if (idx !== -1) {
        const updated = [...prev]
        if (updated[idx].is_service !== 1) {
          updated[idx].quantity += 1
          updated[idx].total = updated[idx].quantity * updated[idx].price
        }
        return updated
      }

      return [
        ...prev,
        {
          product_id: product.id,
          sku: product.sku,
          product_name: displayName,
          quantity: 1,
          price: itemPrice,
          gst_rate: 18,
          total: itemPrice,
          is_service: product.is_service,
          image_url: itemImage,

          v_sku: variantInfo?.v_sku || null,
          variant_name: variantInfo?.variant_name || null,
          color: variantInfo?.color || null,
          size: variantInfo?.size || null,
          v_selling_price: variantInfo?.v_selling_price || null,
          variant_info: variantInfo,
        },
      ]
    })

    toast.success(`${displayName} added`)
    setSkuInput("")
  } catch (err) {
    console.error("SKU scan failed:", err)
    toast.error("Product not found")
  } finally {
    setTimeout(() => {
      scanLockRef.current = false
    }, 500)
  }
}


  useEffect(() => {
    let html5QrCode = null

    if (showCamera && scanMode === "camera") {
      const startScanner = async () => {
        setCameraLoading(true)
        try {
          html5QrCode = new Html5Qrcode("reader")
          scannerRef.current = html5QrCode

          const config = {
            fps: 12,
            qrbox: { width: 250, height: 250 },
            aspectRatio: 1.0,
            formatsToSupport: [
              Html5QrcodeSupportedFormats.QR_CODE,
              Html5QrcodeSupportedFormats.DATA_MATRIX,
              Html5QrcodeSupportedFormats.PDF_417,
              Html5QrcodeSupportedFormats.AZTEC,
              Html5QrcodeSupportedFormats.CODE_128,
              Html5QrcodeSupportedFormats.CODE_39,
              Html5QrcodeSupportedFormats.EAN_13,
            ],
          }

          await html5QrCode.start(
            { facingMode: "environment" },
            config,
            (decodedText) => {
              handleSkuScan(decodedText)

              scannerRef.current?.pause(true)

              setTimeout(() => {
                scannerRef.current?.resume()
              }, 900)
            },
            () => { },
          )
          setCameraLoading(false)
        } catch (err) {
          console.error("Failed to start camera:", err)
          toast.error("Could not access camera")
          setShowCamera(false)
        }
      }

      startScanner()
    }

    return () => {
      if (html5QrCode && html5QrCode.isScanning) {
        html5QrCode
          .stop()
          .then(() => {
            html5QrCode.clear()
          })
          .catch((err) => console.error("Failed to clear scanner", err))
      }
    }
  }, [showCamera])

  const removeLineItem = (index) => {
    setLineItems(lineItems.filter((_, i) => i !== index))
  }

  const calculateTotals = () => {
    const itemsSubtotal = lineItems.reduce((sum, item) => sum + item.total, 0)

    const gstAmount = lineItems.reduce((sum, item) => {
      return sum + (item.total * item.gst_rate) / 100
    }, 0)

    const extraAmount = Number(manualAmount || 0)

    const subtotal = itemsSubtotal + extraAmount
    const total = subtotal + gstAmount - discount

    return { subtotal, gstAmount, total }
  }

  const resetForm = () => {
    setCustomerPhone("")
    setCustomerName("")
    setCustomerEmail("")
    setCustomerAddress("")
    setCustomerId(null)
    setLineItems([])
    setDiscount(0)
    setPaymentStatus("pending")
    setSkuInput("")
    setManualAmount(0)
    setManualLabel("Additional Charge")
    // Reset manual search states
    setProductSearch("")
    setSearchResults([])
    setShowProductSearch(false)
    setSelectedProduct(null)
    setShowVariantDialog(false)
    // Reset new product search states
    setProductSearchQuery("")
    setProductSearchResults([])
    setIsSearchingProducts(false)
    setSelectedProductForVariant(null)
  }
  const handleCreateInvoice = async () => {
    if (!customerName || !customerPhone || lineItems.length === 0) {
      toast.error("Customer name, phone number and at least one item are required")
      return
    }


    // 🚨 SAFETY: Variant products MUST send variant SKU
    for (const item of lineItems) {
      if (item.variant_info && !item.v_sku) {
        toast.error(`Variant SKU missing for ${item.product_name}. Please rescan item.`)
        return
      }
    }

    const { gstAmount } = calculateTotals()

    try {
      const payload = {
        customer_id: customerId,
        customer_name: customerName,
        customer_phone: customerPhone,
        customer_email: customerEmail,
        customer_address: customerAddress,

        items: lineItems.map((item) => ({
          product_id: String(item.product_id),
          product_name: item.product_name,
          quantity: Number(item.quantity),
          price: Number(item.price),
          gst_rate: Number(item.gst_rate || 18),
          total: Number(item.total),
          sku: item.v_sku || item.sku,
          v_sku: item.v_sku,
          variant_name: item.variant_name,
          color: item.color,
          size: item.size,
          // Ensure variant_info is passed correctly if it exists
          variant_info: item.variant_info
            ? {
              v_sku: item.variant_info.v_sku,
              variant_name: item.variant_info.variant_name,
              color: item.variant_info.color,
              size: item.variant_info.size,
              v_selling_price: item.variant_info.v_selling_price,
            }
            : null,
        })),

        gst_amount: gstAmount,
        discount: Number(discount),
        payment_status: paymentStatus,
      }

      await axios.post(`${API}/invoices`, payload)

      toast.success("Invoice created successfully!")
      resetForm()
      fetchInvoices()
      setActiveTab("list")
    } catch (err) {
      console.error("[Invoice Error]", err)
      toast.error(err?.response?.data?.detail || "Failed to create invoice")
    }
  }

  const generatePDF = (invoice) => {
    const printWindow = window.open("", "_blank")
    const { subtotal, gstAmount, total } = calculateInvoiceTotals(invoice)

    const formatProductName = (item) => {
      const name = item.product_name
      const details = []

      if (item.color) details.push(`Color: ${item.color}`)
      if (item.size) details.push(`Size: ${item.size}`)

      // Check variant_info for nested details if top level is missing
      if (!item.color && item.variant_info?.color) details.push(`Color: ${item.variant_info.color}`)
      if (!item.size && item.variant_info?.size) details.push(`Size: ${item.variant_info.size}`)

      if (item.variant_name && !name.includes(item.variant_name)) {
        details.push(item.variant_name)
      }
      if (item.v_sku || item.sku) details.push(`SKU: ${item.v_sku || item.sku}`)

      if (details.length > 0) {
        return `<div style="font-weight: bold;">${name}</div><div style="color: #555; font-size: 10px; margin-top: 2px;">${details.join(" | ")}</div>`
      }

      return name
    }

    const pdfContent = `
      <!DOCTYPE html>
      <html>
      <head>
        <title>Invoice ${invoice.invoice_number}</title>
        <style>
          * { margin: 0; padding: 0; box-sizing: border-box; }
          body { font-family: 'Arial', sans-serif; padding: 40px; color: #333; }
          .invoice-container { max-width: 800px; margin: 0 auto; border: 2px solid #333; padding: 30px; }
          .header { text-align: center; border-bottom: 3px solid #000; padding-bottom: 20px; margin-bottom: 30px; }
          .company-name { font-size: 32px; font-weight: bold; color: #000; margin-bottom: 5px; }
          .company-tagline { font-size: 14px; color: #666; margin-bottom: 10px; }
          .company-details { font-size: 12px; color: #666; }
          .invoice-info { display: flex; justify-content: space-between; margin-bottom: 30px; }
          .info-block { flex: 1; }
          .info-block h3 { font-size: 14px; margin-bottom: 10px; color: #000; border-bottom: 2px solid #000; padding-bottom: 5px; }
          .info-block p { font-size: 12px; margin: 5px 0; }
          table { width: 100%; border-collapse: collapse; margin: 20px 0; }
          thead { background: #000; color: #fff; }
          th, td { padding: 12px; text-align: left; border: 1px solid #ddd; }
          th { font-weight: bold; }
          .text-right { text-align: right; }
          .totals { margin-top: 20px; float: right; width: 300px; }
          .totals table { margin: 0; }
          .totals td { padding: 8px; }
          .totals .total-row { font-weight: bold; font-size: 16px; background: #f0f0f0; }
          .footer { clear: both; margin-top: 50px; padding-top: 20px; border-top: 2px solid #000; }
          .signature { display: flex; justify-content: space-between; margin-top: 40px; }
          .signature-block { text-align: center; }
          .signature-line { border-top: 1px solid #000; width: 200px; margin-top: 50px; padding-top: 5px; }
          .terms { font-size: 10px; color: #666; margin-top: 20px; }
          .badge { display: inline-block; padding: 4px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
          .badge-paid { background: #4CAF50; color: white; }
          .badge-pending { background: #FF9800; color: white; }
          .badge-overdue { background: #F44336; color: white; }
          @media print {
            body { padding: 0; }
            .no-print { display: none; }
          }
        </style>
      </head>
      <body>
        <div class="invoice-container">
          <div class="header">
            <div class="company-name">R RIDE GARAGE</div>
            <div class="company-tagline">Your Trusted Auto Care Partner</div>
            <div class="company-details">
              <p>123 Main Street, Cityname, State - 123456</p>
              <p>Phone: +91 98765 43210 | Email: contact@rridegarage.com</p>
              <p>GSTIN: 29ABCDE1234F1Z5</p>
            </div>
          </div>

          <div class="invoice-info">
            <div class="info-block">
              <h3>INVOICE TO</h3>
              <p><strong>${invoice.customer_name}</strong></p>
              <p>${invoice.customer_phone || "N/A"}</p>
              <p>${invoice.customer_address || "N/A"}</p>
            </div>
            <div class="info-block" style="text-align: right;">
              <h3>INVOICE DETAILS</h3>
              <p><strong>Invoice No:</strong> ${invoice.invoice_number}</p>
              <p><strong>Date:</strong> ${new Date(invoice.created_at).toLocaleDateString()}</p>
              <p><strong>Status:</strong>
                <span class="badge badge-${invoice.payment_status}">
                  ${invoice.payment_status.toUpperCase()}
                </span>
              </p>
            </div>
          </div>

          <table>
            <thead>
              <tr>
                <th>#</th>
                <th>Product</th>
                <th class="text-right">Qty</th>
                <th class="text-right">Price</th>
                <th class="text-right">GST %</th>
                <th class="text-right">Amount</th>
              </tr>
            </thead>
            <tbody>
              ${invoice.items
        .map(
          (item, index) => `
                <tr>
                  <td>${index + 1}</td>
                  <td>${formatProductName(item)}</td>
                  <td class="text-right">${item.quantity}</td>
                  <td class="text-right">₹${formatCurrency(item.price)}</td>
                  <td class="text-right">${item.gst_rate}%</td>
                  <td class="text-right">₹${formatCurrency(item.total)}</td>
                </tr>
              `,
        )
        .join("")}
            </tbody>
          </table>

          <div class="totals">
            <table>
              <tr>
                <td>Subtotal:</td>
                <td class="text-right">₹${formatCurrency(subtotal)}</td>
              </tr>
              <tr>
                <td>GST Amount:</td>
                <td class="text-right">₹${formatCurrency(gstAmount)}</td>
              </tr>
              <tr>
                <td>Discount:</td>
                <td class="text-right">-₹${formatCurrency(invoice.discount)}</td>
              </tr>
              <tr class="total-row">
                <td>Total Amount:</td>
                <td class="text-right">₹${formatCurrency(total)}</td>
              </tr>
            </table>
          </div>

          <div class="footer">
            <div class="terms">
              <h4>Terms & Conditions:</h4>
              <p>1. Payment is due within 30 days of invoice date.</p>
              <p>2. Late payments may incur additional charges.</p>
              <p>3. Goods once sold will not be taken back or exchanged.</p>
              <p>4. All disputes subject to local jurisdiction only.</p>
            </div>

            <div class="signature">
              <div class="signature-block">
                <div class="signature-line">Customer Signature</div>
              </div>
              <div class="signature-block">
                <div class="signature-line">Authorized Signatory</div>
                <p style="margin-top: 10px; font-weight: bold;">R RIDE GARAGE</p>
              </div>
            </div>
          </div>
        </div>

        <div class="no-print" style="text-align: center; margin-top: 20px;">
          <button onclick="window.print()" style="padding: 10px 20px; background: #000; color: #fff; border: none; cursor: pointer; border-radius: 5px;">
            Print Invoice
          </button>
          <button onclick="window.close()" style="padding: 10px 20px; background: #666; color: #fff; border: none; cursor: pointer; border-radius: 5px; margin-left: 10px;">
            Close
          </button>
        </div>
      </body>
      </html>
    `

    printWindow.document.write(pdfContent)
    printWindow.document.close()
  }

  const shareInvoice = async (invoice) => {
    try {
      if (navigator.share) {
        await navigator.share({
          title: `Invoice ${invoice.invoice_number}`,
          text: `Invoice for ${invoice.customer_name} - Total: ₹${invoice.total}`,
          url: window.location.origin, // You can change this to a specific invoice link if available
        })
        toast.success("Shared successfully")
      } else {
        // Fallback to clipboard or just generate PDF
        await navigator.clipboard.writeText(`Invoice ${invoice.invoice_number} for ${invoice.customer_name}`)
        toast.info("Invoice details copied to clipboard (Sharing not supported on this browser)")
        generatePDF(invoice)
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        console.error("Error sharing:", error)
        toast.error("Failed to share")
      }
    }
  }

  const updateInvoiceStatus = async (invoiceId, newStatus) => {
    try {
      await axios.patch(`${API}/invoices/${invoiceId}/status`, null, {
        params: { payment_status: newStatus },
      })

      toast.success("Invoice status updated")
      fetchInvoices()
    } catch (err) {
      toast.error(err?.response?.data?.detail || "Failed to update status")
    }
  }

  const calculateInvoiceTotals = (invoice) => {
    const subtotal = invoice.items.reduce((sum, item) => sum + item.total, 0)
    const gstAmount = invoice.gst_amount
    const total = subtotal + gstAmount - invoice.discount
    return { subtotal, gstAmount, total }
  }

  const filteredInvoices = invoices.filter(
    (inv) =>
      inv.invoice_number.toLowerCase().includes(searchTerm.toLowerCase()) ||
      inv.customer_name.toLowerCase().includes(searchTerm.toLowerCase()) ||
      inv.customer_phone?.toLowerCase().includes(searchTerm.toLowerCase()),
  )

  const { subtotal, gstAmount, total } = calculateTotals()

  const handleSelectProduct = (product) => {
    const variants = product.variants || []

    if (variants.length > 1) {
      // Show variant selection dialog
      setSelectedProductForVariant(product)
      setShowVariantDialog(true)
    } else if (variants.length === 1) {
      // Auto-select the only variant
      addSelectedProductToInvoice(product, variants[0])
    } else {
      // No variants, add base product
      addSelectedProductToInvoice(product, null)
    }

    // Clear search
    setProductSearchQuery("")
    setProductSearchResults([])
  }

  const addSelectedProductToInvoice = (product, variant) => {
    let displayName, itemPrice, itemImage, itemSku, variantInfo

    if (variant) {
      const variantLabel = variant.variant_name || variant.color || variant.size || "Variant"
      displayName = `${product.name} (${variantLabel})`
      itemPrice = variant.v_selling_price || product.selling_price
      itemImage =
        variant.image_url || variant.v_image_url || (product.images && product.images[0]) || "/placeholder.png"
      itemSku = variant.v_sku
      variantInfo = {
        v_sku: variant.v_sku,
        variant_name: variant.variant_name,
        color: variant.color,
        size: variant.size,
        v_selling_price: variant.v_selling_price,
      }
    } else {
      displayName = product.name
      itemPrice = product.selling_price
      itemImage = (product.images && product.images[0]) || "/placeholder.png"
      itemSku = product.sku
      variantInfo = null
    }

    setLineItems((prev) => {
      const key = itemSku
      const idx = prev.findIndex((i) => (i.v_sku || i.sku) === key)

      if (idx !== -1) {
        const updated = [...prev]
        if (updated[idx].is_service !== 1) {
          updated[idx].quantity += 1
          updated[idx].total = updated[idx].quantity * updated[idx].price
        }
        return updated
      }

      return [
        ...prev,
        {
          product_id: product.id,
          sku: product.sku,
          product_name: displayName,
          quantity: 1,
          price: itemPrice,
          gst_rate: 18,
          total: itemPrice,
          is_service: product.is_service,
          image_url: itemImage,
          v_sku: variantInfo?.v_sku || null,
          variant_name: variantInfo?.variant_name || null,
          color: variantInfo?.color || null,
          size: variantInfo?.size || null,
          variant_info: variantInfo,
        },
      ]
    })

    toast.success(`${displayName} added`)
    setShowVariantDialog(false)
    setSelectedProductForVariant(null)
  }

  return (
    <div className="container mx-auto p-4 max-w-6xl pb-20">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3 md:gap-4 mb-6">
        <div>
          <h1 className="text-2xl md:text-3xl lg:text-4xl font-bold">Invoice Management</h1>
          <p className="text-muted-foreground text-sm md:text-base mt-1">Create and manage invoices with GST</p>
        </div>
        <div className="flex gap-2 w-full md:w-auto">
          <Button
            variant={activeTab === "create" ? "default" : "outline"}
            onClick={() => setActiveTab("create")}
            className="flex-1 md:flex-none"
          >
            <Plus className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Create Invoice</span>
            <span className="sm:hidden">Create</span>
          </Button>
          <Button
            variant={activeTab === "list" ? "default" : "outline"}
            onClick={() => setActiveTab("list")}
            className="flex-1 md:flex-none"
          >
            <FileText className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">View Invoices</span>
            <span className="sm:hidden">View</span>
          </Button>
        </div>
      </div>

      {activeTab === "create" && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
          <div className="lg:col-span-2 space-y-6">
            <Card className="border-none shadow-xl bg-card/50 backdrop-blur-sm overflow-hidden">
              <CardHeader className="border-b bg-muted/30">
                <CardTitle className="flex items-center gap-2">
                  <Plus className="w-5 h-5 text-primary" />
                  Create Invoice
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 space-y-6">
                {/* Customer Details Section */}
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                    Customer Details
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                        Customer Phone<span className="text-red-500">*</span>

                      </Label>
                      <Input
                        placeholder="Enter phone number"
                        value={customerPhone}
                        onChange={(e) => setCustomerPhone(e.target.value.replace(/\D/g, ""))}
                        maxLength={10}
                        required
                        className="h-11 rounded-xl"
                      />

                    </div>
                    <div>
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                        Customer Name <span className="text-red-500">*</span>
                      </Label>
                      <Input
                        placeholder="Enter name"
                        value={customerName}
                        onChange={(e) => setCustomerName(e.target.value)}
                        required
                        className="h-11 rounded-xl"
                      />
                    </div>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                        Email
                      </Label>
                     <Input
  type="email"
  placeholder="customer@example.com (optional)"
  value={customerEmail}
  onChange={(e) => setCustomerEmail(e.target.value)}
  className="h-11 rounded-xl"
/>
                    </div>
                    <div>
                      <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                        Address
                      </Label>
                      <Input
                        placeholder="Enter address"
                        value={customerAddress}
                        onChange={(e) => setCustomerAddress(e.target.value)}
                        className="h-11 rounded-xl"
                      />
                    </div>
                  </div>
                </div>

                <div className="space-y-4">
                  <div className="relative">
                    <Label
                      htmlFor="productSearch"
                      className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block"
                    >
                      Search Product by Name or SKU
                    </Label>
                    <div className="relative group">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground group-focus-within:text-primary transition-colors" />
                      <Input
                        id="productSearch"
                        placeholder="Search products (e.g., Thunder, 1309)..."
                        value={productSearchQuery}
                        onChange={(e) => setProductSearchQuery(e.target.value)}
                        className="pl-10 h-12 bg-muted/50 border-muted-foreground/20 focus:border-primary/50 transition-all rounded-xl"
                      />
                      {productSearchQuery && (
                        <button
                          onClick={() => setProductSearchQuery("")}
                          className="absolute right-3 top-1/2 -translate-y-1/2 p-1 rounded-full hover:bg-muted text-muted-foreground"
                        >
                          <X className="w-4 h-4" />
                        </button>
                      )}
                    </div>

                    {/* Search Results Dropdown */}
                    {productSearchResults.length > 0 && (
                      <div className="absolute z-50 w-full mt-2 bg-card border rounded-xl shadow-2xl max-h-[400px] overflow-auto animate-in fade-in zoom-in-95 duration-200">
                        <div className="p-2 space-y-1">
                          {productSearchResults.map((product) => (
                            <button
                              key={product.id}
                              onClick={() => handleSelectProduct(product)}
                              className="w-full flex items-center gap-4 p-3 rounded-lg hover:bg-muted transition-colors text-left group"
                            >
                              <div className="w-12 h-12 rounded-lg bg-muted border flex-shrink-0 overflow-hidden">
                                <img
                                  src={product.images?.[0] || "/placeholder.png"}
                                  alt={product.name}
                                  className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                                />
                              </div>
                              <div className="flex-1 min-w-0">
                                <p className="font-semibold text-sm truncate group-hover:text-primary transition-colors">
                                  {product.name}
                                </p>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                                  <span className="font-mono bg-muted/80 px-1.5 py-0.5 rounded text-[10px]">
                                    {product.sku}
                                  </span>
                                  <span>•</span>
                                  <span className="text-primary font-medium">₹{product.selling_price}</span>
                                  <span>•</span>
                                  {/* <span className={product.stock > 0 ? "text-emerald-500" : "text-red-500"}>
                                    {product.stock > 0 ? `${product.stock} in stock` : "Out"}
                                  </span> */}
                                </div>
                              </div>
                              {product.variants?.length > 0 && (
                                <div className="bg-primary/10 text-primary text-[10px] font-bold px-2 py-1 rounded-full whitespace-nowrap">
                                  {product.variants.length} VARIANTS
                                </div>
                              )}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {isSearchingProducts && (
                      <div className="absolute right-12 top-1/2 -translate-y-1/2">
                        <Loader2 className="w-4 h-4 animate-spin text-primary" />
                      </div>
                    )}
                  </div>

                  {/* <div className="relative">
                    <div className="absolute inset-0 flex items-center">
                      <span className="w-full border-t" />
                    </div>
                    <div className="relative flex justify-center text-xs uppercase">
                      <span className="bg-card px-2 text-muted-foreground">OR SCAN SKU</span>
                    </div>
                  </div>
 */}
                  {/* SKU Scan Section
                  <div className="flex gap-2">
                    <div className="relative flex-1">
                      <ScanLine className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                      <Input
                        ref={skuInputRef}
                        placeholder="Scan or Enter SKU..."
                        value={skuInput}
                        onChange={(e) => setSkuInput(e.target.value)}
                        onKeyDown={(e) => e.key === "Enter" && handleSkuScan(skuInput)}
                        className="pl-10 h-11 bg-muted/50 border-muted-foreground/20 rounded-xl"
                      />
                    </div>
                    <Button
                      onClick={() => handleSkuScan(skuInput)}
                      className="h-11 px-6 rounded-xl shadow-lg shadow-primary/20"
                    >
                      Add
                    </Button>
                  </div> */}
                </div>

                {/* Line Items Section */}
                <div className="space-y-4">
                  <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-2">
                    Line Items
                  </h3>
                  <div className="max-h-[600px] overflow-y-auto pr-2 custom-scrollbar space-y-4">
                    {lineItems.length === 0 ? (
                      <div className="text-center py-10 border-2 border-dashed rounded-2xl border-muted-foreground/20">
                        <Plus className="w-10 h-10 mx-auto mb-2 text-muted-foreground/30" />
                        <p className="text-muted-foreground">Add products to start your invoice</p>
                      </div>
                    ) : (
                      lineItems.map((item, index) => (
                        <div
                          key={index}
                          className="p-4 border rounded-2xl bg-card/50 hover:border-primary/30 transition-all group"
                        >
                          <div className="flex gap-4">
                            <div className="w-20 h-20 rounded-xl bg-muted border overflow-hidden shrink-0">
                              <img
                                src={item.image_url || "/placeholder.png"}
                                alt={item.product_name}
                                className="w-full h-full object-cover"
                              />
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="flex justify-between items-start gap-2">
                                <h4 className="font-bold text-sm line-clamp-2">{item.product_name}</h4>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => removeLineItem(index)}
                                  className="h-8 w-8 text-destructive hover:bg-destructive/10 shrink-0"
                                >
                                  <Trash2 className="w-4 h-4" />
                                </Button>
                              </div>
                              <div className="flex items-center justify-between mt-3">
                                <div className="flex items-center border rounded-lg overflow-hidden h-9">
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    onClick={() => updateLineItem(index, "quantity", Math.max(1, item.quantity - 1))}
                                    className="h-full w-9 rounded-none hover:bg-muted"
                                  >
                                    -
                                  </Button>
                                  <div className="w-10 text-center font-bold text-sm">{item.quantity}</div>
                                  <Button
                                    size="icon"
                                    variant="ghost"
                                    onClick={() => updateLineItem(index, "quantity", item.quantity + 1)}
                                    className="h-full w-9 rounded-none hover:bg-muted"
                                  >
                                    +
                                  </Button>
                                </div>
                                <div className="text-right">
                                  <p className="text-xs text-muted-foreground">Price</p>
                                  <p className="font-bold text-primary">₹{formatCurrency(item.price)}</p>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Sidebar */}
          <div className="lg:col-span-1 space-y-6">
            <Card className="border-none shadow-xl bg-card/50 backdrop-blur-sm overflow-hidden">
              <CardHeader className="border-b bg-muted/30">
                <CardTitle className="flex items-center gap-2">
                  <ScanLine className="w-5 h-5 text-primary" />
                  Scan Item
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 space-y-4">
                {showCamera && (
                  <div className="relative mx-auto max-w-xs overflow-hidden rounded-lg border-2 border-primary aspect-square flex items-center justify-center bg-black">
                    {cameraLoading && (
                      <div className="absolute inset-0 flex items-center justify-center bg-black/50 z-10">
                        <Loader2 className="w-8 h-8 animate-spin text-white" />
                      </div>
                    )}
                    <div id="reader" className="w-full h-full"></div>
                    <div className="absolute bottom-4 left-0 right-0 text-center pointer-events-none z-20">
                      <p className="text-white text-xs bg-black/50 inline-block px-2 py-1 rounded">
                        Align QR code inside the box
                      </p>
                    </div>
                  </div>
                )}

                <div className="flex gap-2">
                  <Button
                    variant={scanMode === "camera" ? "default" : "outline"}
                    onClick={() => {
                      setScanMode("camera")
                      setShowCamera(true)
                      if (scannerRef.current && scannerRef.current.isScanning) {
                        scannerRef.current.resume()
                      }
                    }}
                    className="flex-1 rounded-lg"
                    size="lg"
                  >
                    <Camera className="w-4 h-4 mr-2" /> Use Phone
                  </Button>

                  <Button
                    variant={scanMode === "barcode" ? "default" : "outline"}
                    onClick={() => {
                      setScanMode("barcode")
                      setShowCamera(false)
                      setTimeout(() => skuInputRef.current?.focus(), 200)
                    }}
                    className="flex-1 rounded-lg"
                    size="lg"
                  >
                    <ScanLine className="w-4 h-4 mr-2" /> Use Device
                  </Button>
                </div>

                <div className="relative">
                  <div className="absolute inset-0 flex items-center">
                    <span className="w-full border-t" />
                  </div>
                  <div className="relative flex justify-center text-xs uppercase">
                    <span className="bg-card px-2 text-muted-foreground">OR SCAN SKU</span>
                  </div>
                </div>

                <div className="flex gap-2">
                  <div className="relative flex-1">
                    <ScanLine className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                    <Input
                      ref={skuInputRef}
                      placeholder="Scan or Enter SKU..."
                      value={skuInput}
                      onChange={(e) => setSkuInput(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && handleSkuScan(skuInput)}
                      className="pl-10 h-11 bg-muted/50 border-muted-foreground/20 rounded-xl"
                      disabled={scanMode === "camera"}
                    />
                  </div>
                  <Button
                    onClick={() => handleSkuScan(skuInput)}
                    disabled={scanMode === "camera"}
                    className="h-11 px-6 rounded-xl shadow-lg shadow-primary/20"
                  >
                    Add
                  </Button>
                </div>
              </CardContent>
            </Card>

            <Card className="border-none shadow-xl bg-card/50 backdrop-blur-sm overflow-hidden">
              <CardHeader className="border-b bg-muted/30">
                <CardTitle className="flex items-center gap-2">
                  <ScanLine className="w-5 h-5 text-primary" />
                  Invoice Summary
                </CardTitle>
              </CardHeader>
              <CardContent className="p-6 space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                      Additional Amount
                    </Label>
                    <Input
                      type="number"
                      min="0"
                      placeholder="Enter amount"
                      value={manualAmount}
                      onChange={(e) => setManualAmount(Number(e.target.value))}
                      className="h-11 rounded-xl"
                    />
                  </div>

                  <div>
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                      Description
                    </Label>
                    <Input
                      placeholder="Eg: Labour / Fitting / Service"
                      value={manualLabel}
                      onChange={(e) => setManualLabel(e.target.value)}
                      className="h-11 rounded-xl"
                    />
                  </div>

                  <div>
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                      Discount (₹)
                    </Label>
                    <Input
                      type="number"
                      value={discount}
                      onChange={(e) => setDiscount(Number(e.target.value))}
                      min="0"
                      className="h-11 rounded-xl"
                    />
                  </div>
                  <div>
                    <Label className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2 block">
                      Payment Status
                    </Label>
                    <Select value={paymentStatus} onValueChange={setPaymentStatus}>
                      <SelectTrigger className="h-11 rounded-xl">
                        <SelectValue placeholder="Select status" />
                      </SelectTrigger>

                      <SelectContent className="bg-popover text-popover-foreground border-border shadow-md rounded-xl">
                        <SelectItem value="pending" className="focus:bg-accent focus:text-accent-foreground">
                          Pending
                        </SelectItem>
                        <SelectItem value="paid" className="focus:bg-accent focus:text-accent-foreground">
                          Paid
                        </SelectItem>
                        <SelectItem value="overdue" className="focus:bg-accent focus:text-accent-foreground">
                          Overdue
                        </SelectItem>
                        <SelectItem value="cancelled" className="focus:bg-accent focus:text-accent-foreground">
                          Cancelled
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>

                <div className="border-t pt-4 space-y-2 text-sm">
                  <div className="flex justify-between text-muted-foreground">
                    <span>Subtotal:</span>
                    <span className="font-medium text-foreground">₹{formatCurrency(subtotal)}</span>
                  </div>
                  {manualAmount > 0 && (
                    <div className="flex justify-between text-muted-foreground">
                      <span>{manualLabel}:</span>
                      <span className="font-medium text-foreground">₹{formatCurrency(manualAmount)}</span>
                    </div>
                  )}

                  <div className="flex justify-between text-muted-foreground">
                    <span>GST Amount:</span>
                    <span className="font-medium text-foreground">₹{formatCurrency(gstAmount)}</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>Discount:</span>
                    <span className="font-medium text-foreground">-₹{formatCurrency(discount)}</span>
                  </div>
                  <div className="flex justify-between font-bold text-base border-t pt-3 mt-2">
                    <span>Total:</span>
                    <span>₹{formatCurrency(total)}</span>
                  </div>
                </div>

                <Button
                  className="w-full mt-6 h-12 rounded-xl shadow-lg shadow-primary/20"
                  onClick={handleCreateInvoice}
                  disabled={loading}
                >
                  {loading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    "Create Invoice"
                  )}
                </Button>
              </CardContent>
            </Card>
          </div>
        </div>
      )}

      {activeTab === "list" && (
        <div className="space-y-4">
          <div className="relative">
            <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search by invoice number, customer name, or phone..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="pl-9 h-11 rounded-xl"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
            <Select
              onValueChange={(v) => {
                setPage(1)
                setStatusFilter(v)
              }}
            >
              <SelectTrigger className="h-11 rounded-xl">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border shadow-md rounded-xl">
                <SelectItem value="paid" className="focus:bg-accent focus:text-accent-foreground">
                  Paid
                </SelectItem>
                <SelectItem value="overdue" className="focus:bg-accent focus:text-accent-foreground">
                  Overdue
                </SelectItem>
                <SelectItem value="pending" className="focus:bg-accent focus:text-accent-foreground">
                  Pending
                </SelectItem>
                <SelectItem value="cancelled" className="focus:bg-accent focus:text-accent-foreground">
                  Cancelled
                </SelectItem>
              </SelectContent>
            </Select>

            <Select
              onValueChange={(v) => {
                setPage(1)
                setRangeFilter(v)
              }}
            >
              <SelectTrigger className="h-11 rounded-xl">
                <SelectValue placeholder="Date Range" />
              </SelectTrigger>
              <SelectContent className="bg-popover text-popover-foreground border-border shadow-md rounded-xl">
                <SelectItem value="last10" className="focus:bg-accent focus:text-accent-foreground">
                  Last 10 Days
                </SelectItem>
                <SelectItem value="last30" className="focus:bg-accent focus:text-accent-foreground">
                  Last 30 Days
                </SelectItem>
              </SelectContent>
            </Select>

            <Input
              type="month"
              value={monthFilter}
              onChange={(e) => {
                setPage(1)
                setMonthFilter(e.target.value)
              }}
              className="h-11 rounded-xl col-span-1 sm:col-span-2 md:col-span-1"
            />
          </div>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg md:text-xl">All Invoices ({filteredInvoices.length})</CardTitle>
            </CardHeader>
            <CardContent>
              {/* Desktop */}
              <div className="hidden md:block overflow-x-auto">
                <table className="w-full border-collapse">
                  <thead>
                    <tr className="border-b text-sm text-muted-foreground">
                      <th className="p-3 text-left">Invoice #</th>
                      <th className="p-3 text-left">Date</th>
                      <th className="p-3 text-left">Customer</th>
                      <th className="p-3 text-left">Phone</th>
                      <th className="p-3 text-right">Total</th>
                      <th className="p-3 text-left">Status</th>
                      <th className="p-3 text-right">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredInvoices
                      .filter((inv) => inv && inv.id)
                      .map((inv) => (
                        <tr key={inv.id} className="border-b hover:bg-muted/40 transition-colors">
                          <td className="p-3 font-mono text-sm">{inv.invoice_number}</td>
                          <td className="p-3 text-sm">{new Date(inv.created_at).toLocaleDateString()}</td>
                          <td className="p-3 font-medium">{inv.customer_name}</td>
                          <td className="p-3 text-sm">{inv.customer_phone || "N/A"}</td>
                          <td className="p-3 text-right font-semibold">₹{formatCurrency(inv.total)}</td>
                          <td className="p-3">
                            <Select
                              value={inv.payment_status || "pending"}
                              onValueChange={(v) => {
                                updateInvoiceStatus(inv.id, v)
                                setInvoices((prev) =>
                                  prev.map((i) => (i.id === inv.id ? { ...i, payment_status: v } : i)),
                                )
                              }}
                            >
                              <SelectTrigger className="w-36 bg-black text-white border-gray-700 rounded-lg">
                                <SelectValue placeholder="Status" />
                              </SelectTrigger>

                              <SelectContent className="bg-popover text-popover-foreground border-border shadow-md rounded-xl">
                                <SelectItem value="pending" className="focus:bg-accent focus:text-accent-foreground">
                                  Pending
                                </SelectItem>
                                <SelectItem value="paid" className="focus:bg-accent focus:text-accent-foreground">
                                  Paid
                                </SelectItem>
                                <SelectItem value="overdue" className="focus:bg-accent focus:text-accent-foreground">
                                  Overdue
                                </SelectItem>
                                <SelectItem value="cancelled" className="focus:bg-accent focus:text-accent-foreground">
                                  Cancelled
                                </SelectItem>
                              </SelectContent>
                            </Select>
                          </td>

                          <td className="p-3 text-right">
                            <div className="flex items-center justify-end gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => setViewInvoice(inv)}
                                className="rounded-lg"
                              >
                                <Eye className="w-4 h-4 mr-1" /> View
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                onClick={() => generatePDF(inv)}
                                className="rounded-lg"
                              >
                                <Download className="w-4 h-4 mr-1" /> PDF
                              </Button>
                              {/* <Button
                                size="sm"
                                variant="outline"
                                onClick={() => shareInvoice(inv)}
                                className="rounded-lg"
                              >
                                <Share2 className="w-4 h-4 mr-1" /> Share
                              </Button> */}
                            </div>
                          </td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>

              {/* Mobile */}
              <div className="md:hidden space-y-3">
                {filteredInvoices
                  .filter((inv) => inv && inv.id)
                  .map((inv) => (
                    <Card key={inv.id} className="border shadow-sm hover:shadow-md transition-shadow rounded-xl">
                      <CardContent className="p-4 space-y-3">
                        <div className="flex items-start justify-between gap-3">
                          <div className="space-y-1 min-w-0 flex-1">
                            <p className="font-mono text-sm font-semibold truncate">{inv.invoice_number}</p>
                            <p className="text-xs text-muted-foreground">
                              {new Date(inv.created_at).toLocaleDateString()}
                            </p>
                          </div>
                          <div className="text-right shrink-0">
                            <p className="text-lg font-bold">₹{formatCurrency(inv.total)}</p>
                          </div>
                        </div>

                        <div className="space-y-1 border-t pt-3">
                          <p className="font-medium text-sm">{inv.customer_name}</p>
                          <p className="text-xs text-muted-foreground">{inv.customer_phone || "N/A"}</p>
                        </div>

                        <div className="space-y-2 border-t pt-3">
                          <Label className="text-xs font-medium uppercase tracking-wider text-muted-foreground block">
                            Payment Status
                          </Label>
                          <Select
                            value={inv.payment_status || "pending"}
                            onValueChange={(v) => {
                              updateInvoiceStatus(inv.id, v)
                              setInvoices((prev) =>
                                prev.map((i) => (i.id === inv.id ? { ...i, payment_status: v } : i)),
                              )
                            }}
                          >
                            <SelectTrigger className="rounded-lg">
                              <SelectValue placeholder="Status" />
                            </SelectTrigger>

                            <SelectContent className="bg-popover text-popover-foreground border-border shadow-md rounded-xl">
                              <SelectItem value="pending" className="focus:bg-accent focus:text-accent-foreground">
                                Pending
                              </SelectItem>
                              <SelectItem value="paid" className="focus:bg-accent focus:text-accent-foreground">
                                Paid
                              </SelectItem>
                              <SelectItem value="overdue" className="focus:bg-accent focus:text-accent-foreground">
                                Overdue
                              </SelectItem>
                              <SelectItem value="cancelled" className="focus:bg-accent focus:text-accent-foreground">
                                Cancelled
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="flex gap-2 pt-2 border-t">
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => setViewInvoice(inv)}
                            className="flex-1 rounded-lg"
                          >
                            <Eye className="w-4 h-4 mr-2" /> View
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => generatePDF(inv)}
                            className="flex-1 rounded-lg"
                          >
                            <Download className="w-4 h-4 mr-2" /> PDF
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            onClick={() => shareInvoice(inv)}
                            className="flex-1 rounded-lg"
                          >
                            <Share2 className="w-4 h-4 mr-2" /> Share
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
              </div>

              {filteredInvoices.length === 0 && (
                <div className="py-12 text-center text-muted-foreground">
                  <FileText className="w-12 h-12 mx-auto mb-3 opacity-20" />
                  <p className="text-sm">No invoices found</p>
                </div>
              )}

              <div className="flex flex-col sm:flex-row justify-between items-center gap-3 mt-6 pt-4 border-t">
                <Button
                  variant="outline"
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                  className="w-full sm:w-auto rounded-lg"
                  size="sm"
                >
                  Previous
                </Button>

                <span className="text-sm font-medium">
                  Page {pagination.page} of {pagination.total_pages}
                </span>

                <Button
                  variant="outline"
                  disabled={page === pagination.total_pages}
                  onClick={() => setPage(page + 1)}
                  className="w-full sm:w-auto rounded-lg"
                  size="sm"
                >
                  Next
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {viewInvoice && (
        <Dialog open={!!viewInvoice} onOpenChange={() => setViewInvoice(null)}>
          <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto rounded-3xl p-0">
            <DialogHeader className="bg-primary p-6 text-primary-foreground">
              <DialogTitle className="text-xl font-bold flex items-center gap-2">
                <FileText className="w-5 h-5" /> Invoice Details
              </DialogTitle>
              <p className="text-primary-foreground/80 text-sm mt-1">Invoice #{viewInvoice.invoice_number}</p>
            </DialogHeader>
            <div className="p-6 space-y-6">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Customer Name</p>
                  <p className="font-medium mt-1 text-base">{viewInvoice.customer_name}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Phone</p>
                  <p className="font-medium mt-1 text-base">{viewInvoice.customer_phone || "N/A"}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Date</p>
                  <p className="font-medium mt-1 text-base">{new Date(viewInvoice.created_at).toLocaleDateString()}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase tracking-wider font-semibold">Payment Status</p>
                  <p className="font-medium mt-1 capitalize text-base">{viewInvoice.payment_status}</p>
                </div>
              </div>

              <div className="border rounded-lg p-4">
                <h4 className="font-bold mb-3 text-sm uppercase tracking-wider text-muted-foreground">Items</h4>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b text-muted-foreground text-xs uppercase tracking-wider">
                        <th className="p-2 text-left">Product</th>
                        <th className="p-2 text-right">Qty</th>
                        <th className="p-2 text-right">Price</th>
                        <th className="p-2 text-right">Total</th>
                      </tr>
                    </thead>
                    <tbody>
                      {viewInvoice.items.map((item, idx) => (
                        <tr key={idx} className="border-b">
                          <td className="p-2 align-top">
                            <div className="font-medium">{item.product_name}</div>
                            {(item.color || item.size || item.v_sku || item.variant_info) && (
                              <div className="text-[10px] text-muted-foreground flex flex-wrap gap-x-2 mt-0.5">
                                {(item.color || item.variant_info?.color) && (
                                  <span>Color: {item.color || item.variant_info.color}</span>
                                )}
                                {(item.size || item.variant_info?.size) && (
                                  <span>Size: {item.size || item.variant_info.size}</span>
                                )}
                                {(item.v_sku || item.sku || item.variant_info?.v_sku) && (
                                  <span className="font-mono bg-muted px-1 rounded">
                                    SKU: {item.v_sku || item.sku || item.variant_info?.v_sku}
                                  </span>
                                )}
                              </div>
                            )}
                          </td>
                          <td className="text-right p-2 align-top">{item.quantity}</td>
                          <td className="text-right p-2 align-top">₹{formatCurrency(item.price)}</td>
                          <td className="text-right p-2 align-top font-semibold">₹{formatCurrency(item.total)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="mt-4 space-y-2 text-sm">
                  <div className="flex justify-between text-muted-foreground">
                    <span>Subtotal:</span>
                    <span className="font-medium text-foreground">₹{formatCurrency(viewInvoice.subtotal)}</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>GST:</span>
                    <span className="font-medium text-foreground">₹{formatCurrency(viewInvoice.gst_amount)}</span>
                  </div>
                  <div className="flex justify-between text-muted-foreground">
                    <span>Discount:</span>
                    <span className="font-medium text-foreground">-₹{formatCurrency(viewInvoice.discount)}</span>
                  </div>
                  <div className="flex justify-between font-bold text-base border-t pt-2 mt-2">
                    <span>Total:</span>
                    <span>₹{formatCurrency(viewInvoice.total)}</span>
                  </div>
                </div>
              </div>

              <Button
                className="w-full h-12 rounded-xl shadow-lg shadow-primary/20"
                onClick={() => generatePDF(viewInvoice)}
              >
                <Download className="w-4 h-4 mr-2" />
                Download PDF
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      )}

      <Dialog open={showVariantDialog} onOpenChange={setShowVariantDialog}>
        <DialogContent className="sm:max-w-[500px] rounded-3xl overflow-hidden p-0 border-none shadow-2xl">
          <DialogHeader className="bg-primary p-6 text-primary-foreground">
            <DialogTitle className="text-xl font-bold flex items-center gap-2">Select Variant</DialogTitle>
            <p className="text-primary-foreground/80 text-sm mt-1">
              {selectedProductForVariant?.name} has multiple options. Please select one.
            </p>
          </DialogHeader>

          <div className="p-6 max-h-[60vh] overflow-auto">
            <div className="grid grid-cols-1 gap-3">
              {selectedProductForVariant?.variants?.map((variant) => (
                <button
                  key={variant.v_sku}
                  onClick={() => addSelectedProductToInvoice(selectedProductForVariant, variant)}
                  className="flex items-center gap-4 p-4 rounded-2xl border bg-card hover:border-primary hover:shadow-lg transition-all text-left group"
                >
                  <div className="w-16 h-16 rounded-xl bg-muted border overflow-hidden flex-shrink-0">
                    <img
                      src={
                        variant.image_url ||
                        variant.v_image_url ||
                        selectedProductForVariant.images?.[0] ||
                        "/placeholder.png" ||
                        "/placeholder.svg" ||
                        "/placeholder.svg"
                      }
                      alt={variant.variant_name}
                      className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-300"
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="font-bold text-sm">
                      {variant.variant_name ||
                        `${variant.color || ""} ${variant.size || ""}`.trim() ||
                        "Default Variant"}
                    </p>
                    <p className="text-xs text-muted-foreground font-mono mt-0.5">{variant.v_sku}</p>
                    <div className="flex items-center gap-3 mt-2">
                      <span className="text-primary font-bold">
                        ₹{variant.v_selling_price || selectedProductForVariant.selling_price}
                      </span>
                      <span className="text-[10px] text-muted-foreground bg-muted px-2 py-0.5 rounded-full">
                        Stock: {variant.stock}
                      </span>
                    </div>
                  </div>
                  <div className="w-10 h-10 rounded-full bg-primary/5 flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
                    <Plus className="w-5 h-5" />
                  </div>
                </button>
              ))}
            </div>
          </div>
          <DialogFooter className="p-4 bg-muted/30 border-t flex justify-end">
            <Button variant="ghost" onClick={() => setShowVariantDialog(false)} className="rounded-xl">
              Cancel
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
