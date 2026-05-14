"use client"

import { useEffect, useState, useRef } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogFooter } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetFooter } from "@/components/ui/sheet"
import { toast } from "sonner"
import { Plus, Trash2, Search, Download, Eye, FileText, ScanLine, Loader2, Camera, X, Share2, Menu, Home, CheckCircle2, Minus, Wallet, Receipt, Landmark } from "lucide-react"
import { Html5Qrcode, Html5QrcodeSupportedFormats } from "html5-qrcode"
import { Capacitor } from "@capacitor/core"
import { useLocation, useNavigate } from "react-router-dom"
import {
    initializePdfDownloadSupport,
    saveFileWithNotification,
    savePdfWithNotification,
} from "@/lib/pdfHandler"
import { buildInvoicePrintHtml } from "@/lib/invoicePrintTemplate"

const API = `${process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"}/api`

const RUPEE_SYMBOL = "\u20B9"

const formatCurrency = (amount, decimals = 2) => {
    const num = typeof amount === "number" ? amount : Number(amount)
    return isNaN(num) ? "0.00" : num.toFixed(decimals)
}

const formatDisplayCurrency = (amount, decimals = 2) =>
    `${RUPEE_SYMBOL}\u00A0${formatCurrency(amount, decimals)}`

const generateManualSku = () =>
    `MANUAL-${Date.now()}-${Math.random().toString(36).slice(2, 8).toUpperCase()}`

const formatNumber = (amount, decimals = 0) => {
    const num = typeof amount === "number" ? amount : Number(amount)
    return isNaN(num) ? "0" : num.toFixed(decimals)
}

const normalizePhoneNumber = (value) => value.replace(/\D/g, "").slice(0, 10)

const normalizeCustomerBalance = (value) => {
    const num = typeof value === "number" ? value : Number(value)
    if (isNaN(num)) return 0
    return Math.abs(num) < 0.01 ? 0 : num
}

const resolveInvoicePdfUrl = (pdfUrl) => {
    if (!pdfUrl) return null
    if (/^https?:\/\//i.test(pdfUrl)) return pdfUrl
    const backendBase = process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"
    return `${backendBase}${pdfUrl.startsWith("/") ? "" : "/"}${pdfUrl}`
}

const getInvoicePdfPath = (invoice) => {
    if (!invoice?.id) return invoice?.pdf_url || null
    return invoice?.pdf_url || `/api/invoices/${invoice.id}/pdf`
}

const getInvoiceDisplayTotal = (invoice) =>
    Number(invoice?.original_total || invoice?.display_total || invoice?.total || 0)

const getPendingAmount = (balance) => Math.max(normalizeCustomerBalance(balance), 0)

const getAdvanceAmount = (balance) => Math.max(-normalizeCustomerBalance(balance), 0)

const getCustomerBalanceText = (balance) => {
    const pending = getPendingAmount(balance)
    if (pending > 0) {
    return `Pending balance: ${formatDisplayCurrency(pending)}`
    }

    const advance = getAdvanceAmount(balance)
    if (advance > 0) {
    return `Advance available: ${formatDisplayCurrency(advance)}`
    }

    return "No pending or advance balance"
}

const PAYMENT_STATUS_VALUES = new Set(["pending", "paid", "partial", "cancelled"])
const PAYMENT_MODE_VALUES = new Set(["cash", "upi", "bank", "cheque"])

const normalizePaymentStatus = (value, fallback = "pending") => {
    const normalized = String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z]/g, "")

    return PAYMENT_STATUS_VALUES.has(normalized) ? normalized : fallback
}

const normalizePaymentMode = (value, fallback = "upi") => {
    const normalized = String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z]/g, "")

    if (PAYMENT_MODE_VALUES.has(normalized)) {
        return normalized
    }

    if (normalized === "card") {
        return "bank"
    }

    return fallback
}

const invoiceMatchesSearch = (invoice, rawSearch) => {
    const query = String(rawSearch || "").trim().toLowerCase()
    if (!query) return true

    const baseFields = [
        invoice?.invoice_number,
        invoice?.customer_name,
        invoice?.customer_phone,
        invoice?.customer_address,
        invoice?.created_by,
    ]

    if (baseFields.some((field) => String(field || "").toLowerCase().includes(query))) {
        return true
    }

    const items = Array.isArray(invoice?.items) ? invoice.items : []

    return items.some((item) => {
        const variant = item?.variant_info || {}
        return [
            item?.product_name,
            item?.sku,
            item?.v_sku,
            item?.variant_name,
            item?.color,
            item?.size,
            variant?.v_sku,
            variant?.variant_name,
            variant?.color,
            variant?.size,
        ].some((field) => String(field || "").toLowerCase().includes(query))
    })
}

const blobToBase64 = (blob) =>
    new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onloadend = () => {
            const result = reader.result
            if (typeof result !== "string") {
                reject(new Error("Failed to read PDF file"))
                return
            }
            resolve(result.split(",")[1] || "")
        }
        reader.onerror = () => reject(new Error("Failed to convert PDF"))
        reader.readAsDataURL(blob)
    })

export default function InvoiceCreateCompact() {
    const location = useLocation()
    const navigate = useNavigate()
    const [activeTab, setActiveTab] = useState(
        location.state?.tab || "create"
    )
    const resultRefs = useRef([])
    const [previewImage, setPreviewImage] = useState(null)
    const [invoices, setInvoices] = useState([])
    const [products, setProducts] = useState([])
    const [open, setOpen] = useState(false)
    const [viewInvoice, setViewInvoice] = useState(null)
    const [loading, setLoading] = useState(false)
    const [searchTerm, setSearchTerm] = useState("")
    const [showMobileFilters, setShowMobileFilters] = useState(false)
    const [page, setPage] = useState(1)
    const [statusFilter, setStatusFilter] = useState("all")
    const [rangeFilter, setRangeFilter] = useState("last30")
    const [monthFilter, setMonthFilter] = useState(null)
    const [exportDialogOpen, setExportDialogOpen] = useState(false)
    const [exportRange, setExportRange] = useState("today")
    const [exportingInvoices, setExportingInvoices] = useState(false)
    const [todayStats, setTodayStats] = useState({
        total_sales_today: 0,
        invoices_today: 0,
        cash_sales_today: 0,
        online_sales_today: 0,
    })

const isNativePlatform = () => Capacitor.isNativePlatform()

    const [limit, setLimit] = useState(10)
    const [pagination, setPagination] = useState({ page: 1, total_pages: 1 })
    const [discount, setDiscount] = useState("")
    const [paymentStatus, setPaymentStatus] = useState("pending")
    const [paidAmount, setPaidAmount] = useState(0)
    const [combos, setCombos] = useState([])
    const [comboModal, setComboModal] = useState(false)
    const [drafts, setDrafts] = useState([])
    const [paymentMode, setPaymentMode] = useState("upi")
    const [productSearch, setProductSearch] = useState("")
    const [searchResults, setSearchResults] = useState([])
    const [showProductSearch, setShowProductSearch] = useState(false)
    const [selectedProduct, setSelectedProduct] = useState(null)
    const [showVariantDialog, setShowVariantDialog] = useState(false)
    const [creatingInvoice, setCreatingInvoice] = useState(false)
    const [selectedVariantIndex, setSelectedVariantIndex] = useState(0)
    const [gstEnabled, setGstEnabled] = useState(false)
    const [gstRate, setGstRate] = useState(18)
    const INVOICE_STORAGE_KEY = "invoice_draft_state"
    const draftToastLock = useRef(false)
    const hasRestoredDraftRef = useRef(false)
    const draftSaveQueueRef = useRef(Promise.resolve())
    const draftIdRef = useRef(null)
    const [showManualDialog, setShowManualDialog] = useState(false)
    const [manualItem, setManualItem] = useState({ name: "", price: "", quantity: 1 })

    const [customerPhone, setCustomerPhone] = useState("")
    const [customerName, setCustomerName] = useState("")
    const [customerEmail, setCustomerEmail] = useState("")
    const [customerAddress, setCustomerAddress] = useState("")
    const [customerId, setCustomerId] = useState(null)
    const [customerBalance, setCustomerBalance] = useState(0)
    const [useAdvance, setUseAdvance] = useState(false)
    const [draftId, setDraftId] = useState(null)
    const [isDraft, setIsDraft] = useState(false)

    const [lineItems, setLineItems] = useState([])
    const [skuInput, setSkuInput] = useState("")
    const [isScanning, setIsScanning] = useState(false)
    const [showCamera, setShowCamera] = useState(false)
    const [cameraLoading, setCameraLoading] = useState(false)
    const [lastScannedSku, setLastScannedSku] = useState(null)
    const [lastScanTime, setLastScanTime] = useState(0)
    const scannerRef = useRef(null)
    const scanLockRef = useRef(false)
    const [scanMode, setScanMode] = useState(null)
    const skuInputRef = useRef(null)
    const scanTimeoutRef = useRef(null)

    const [productSearchQuery, setProductSearchQuery] = useState("")
    const [productSearchResults, setProductSearchResults] = useState([])
    const [isSearchingProducts, setIsSearchingProducts] = useState(false)
    const searchTimeoutRef = useRef(null)
    const [selectedProductForVariant, setSelectedProductForVariant] = useState(null)
    const [selectedSearchIndex, setSelectedSearchIndex] = useState(-1)
    const [pendingCombo, setPendingCombo] = useState(null)
    const [additionalCharges, setAdditionalCharges] = useState([{ label: "", amount: "" }])

    const [isPaymentModalOpen, setIsPaymentModalOpen] = useState(false)
    const [paymentHistory, setPaymentHistory] = useState([])
    const [paymentAmount, setPaymentAmount] = useState("")
    const [paymentModeSelected, setPaymentModeSelected] = useState("upi")
    const [paymentReference, setPaymentReference] = useState("")
    const [loadingPayments, setLoadingPayments] = useState(false)
    const [sharingInvoiceId, setSharingInvoiceId] = useState(null)
    const [isPaidModeDialogOpen, setIsPaidModeDialogOpen] = useState(false)
    const [paidPaymentMode, setPaidPaymentMode] = useState("upi")
    const [paidPaymentRef, setPaidPaymentRef] = useState("")
    const [pendingPaidInvoiceId, setPendingPaidInvoiceId] = useState(null)

    // NEW: Payment drawer state
    const [showPaymentDrawer, setShowPaymentDrawer] = useState(false)

    // NEW: Success modal state
    const [showSuccessModal, setShowSuccessModal] = useState(false)
    const [createdInvoice, setCreatedInvoice] = useState(null)

    const { subtotal, gstAmount, total } = calculateTotals()
    const pendingBalance = getPendingAmount(customerBalance)
    const advanceAvailable = pendingBalance > 0 ? 0 : getAdvanceAmount(customerBalance)
    const appliedAdvance = useAdvance ? Math.min(advanceAvailable, Math.max(total, 0)) : 0
    const payableTotal = Math.max(total - appliedAdvance, 0)
    const isAdvanceCovered = appliedAdvance > 0 && payableTotal <= 0
    const normalizedPaymentStatus = normalizePaymentStatus(paymentStatus)
    const normalizedPaymentMode = normalizePaymentMode(paymentMode)
    const effectivePaymentStatus = isAdvanceCovered
        ? "paid"
        : (
            normalizedPaymentStatus === "paid"
                ? "paid"
                : (normalizedPaymentStatus === "partial" || appliedAdvance > 0
                    ? "partial"
                    : normalizedPaymentStatus)
        )
    const remainingBalance = Math.max(
        effectivePaymentStatus === "partial"
            ? payableTotal - Number(paidAmount || 0)
            : effectivePaymentStatus === "pending"
                ? payableTotal
                : 0,
        0
    )

    // ============= HELPER FUNCTIONS =============

    function calculateTotals() {
        const safeItems = Array.isArray(lineItems) ? lineItems : []
        const itemsSubtotal = safeItems.reduce((sum, item) => sum + Number(item.total || 0), 0)
        const extraAmount = additionalCharges.reduce((sum, c) => sum + Number(c.amount || 0), 0)
        const taxableAmount = itemsSubtotal + extraAmount
        const gstAmount = gstEnabled ? (taxableAmount * Number(gstRate || 0)) / 100 : 0
        const total = taxableAmount - Number(discount || 0) + gstAmount

        return {
            subtotal: itemsSubtotal,
            additionalTotal: extraAmount,
            taxableAmount,
            gstAmount,
            total
        }
    }

    const buildDraftPayload = () => {
        const cleanedItems = lineItems
            .filter((item) => item && item.product_name && Number(item.quantity) > 0)
            .map((item) => ({
                product_id: item.product_id || null,
                combo_id: item.combo_id || null,
                product_name: item.product_name,
                quantity: Number(item.quantity),
                price: Number(item.price),
                gst_rate: gstEnabled ? Number(gstRate) : 0,
                total: Number(item.total),
                sku: item.v_sku || item.sku || null,
                is_service: item.is_service || 0,
                variant_info: item.variant_info || null,
                image_url: item.image_url || null
            }))

        const cleanedAdditionalCharges = additionalCharges.filter(
            (charge) => charge.label?.trim() || Number(charge.amount) > 0
        )

        return {
            customer_id: customerId,
            customer_name: customerName,
            customer_phone: customerPhone,
            customer_email: customerEmail,
            customer_address: customerAddress,
            items: cleanedItems,
            gst_enabled: gstEnabled,
            gst_rate: gstEnabled ? Number(gstRate) : 0,
            discount: Number(discount || 0),
            additional_charges: cleanedAdditionalCharges,
            payment_status: effectivePaymentStatus,
            payment_mode: normalizedPaymentMode,
            paid_amount: !isAdvanceCovered && effectivePaymentStatus !== "pending" ? Number(paidAmount || 0) : 0,
            use_advance: Boolean(useAdvance && appliedAdvance > 0)
        }
    }

    const persistDraft = async (payload) => {
        const runSave = async () => {
            let currentDraftId = draftIdRef.current || draftId
            let createdNewDraft = false

            if (currentDraftId) {
                try {
                    const res = await axios.put(`${API}/invoices/draft/${currentDraftId}`, payload)
                    const savedDraftId = res?.data?.id || currentDraftId
                    draftIdRef.current = savedDraftId
                    if (savedDraftId !== draftId) {
                        setDraftId(savedDraftId)
                    }
                    setIsDraft(true)
                    return { id: savedDraftId, createdNewDraft: false }
                } catch (err) {
                    if (err?.response?.status !== 404) {
                        throw err
                    }
                }
            }

            const res = await axios.post(`${API}/invoices/draft`, payload)
            currentDraftId = res.data.id
            createdNewDraft = true
            draftIdRef.current = currentDraftId
            setDraftId(currentDraftId)
            setIsDraft(true)
            return { id: currentDraftId, createdNewDraft }
        }

        draftSaveQueueRef.current = draftSaveQueueRef.current
            .catch(() => undefined)
            .then(runSave)

        return draftSaveQueueRef.current
    }

    const clearDraftPersistence = () => {
        draftIdRef.current = null
        setDraftId(null)
        setIsDraft(false)
        localStorage.removeItem(INVOICE_STORAGE_KEY)
    }

    const addAdditionalCharge = () => {
        setAdditionalCharges([...additionalCharges, { label: "", amount: "" }])
    }

    const removeAdditionalCharge = (index) => {
        const updated = [...additionalCharges]
        updated.splice(index, 1)
        setAdditionalCharges(updated)
    }

    const updateAdditionalCharge = (index, field, value) => {
        const updated = [...additionalCharges]
        updated[index][field] = value
        setAdditionalCharges(updated)
    }

    const additionalTotal = additionalCharges
        .filter(c => c.label || c.amount)
        .reduce((sum, c) => sum + Number(c.amount || 0), 0)

    const addManualItemToInvoice = () => {
        if (!manualItem.name || !manualItem.price) {
            toast.error("Item name and price required")
            return
        }

        const qty = Number(manualItem.quantity || 1)
        const price = Number(manualItem.price || 0)

        const newItem = {
            product_id: null,
            sku: generateManualSku(),
            product_name: manualItem.name,
            quantity: qty,
            price: price,
            gst_rate: gstEnabled ? Number(gstRate) : 0,
            total: qty * price,
            is_service: 1,
            image_url: "/placeholder.png",
            variant_info: null,
        }

        setLineItems((prev) => [...prev, newItem])
        setManualItem({ name: "", price: "", quantity: 1 })
        setShowManualDialog(false)
        toast.success("Manual item added")
    }

    const handleCreateInvoice = async () => {
        if (creatingInvoice) return

        if (!customerName || !customerPhone || lineItems.length === 0) {
            toast.error("Customer name, phone number and items required")
            return
        }

        if (customerPhone.length !== 10) {
            toast.error("Enter a valid 10-digit phone number")
            return
        }

        if (useAdvance && pendingBalance > 0) {
            toast.error("Advance cannot be used while the customer has pending balance")
            return
        }

        if (!isAdvanceCovered && !paymentStatus) {
            toast.error("Payment status is required")
            return
        }

        if (!isAdvanceCovered && Number(paidAmount || 0) > 0 && !paymentMode) {
            toast.error("Payment mode required")
            return
        }

        if (!isAdvanceCovered && effectivePaymentStatus === "partial") {
            if (Number(paidAmount || 0) < 0) {
                toast.error("Paid amount cannot be negative")
                return
            }

            if (Number(paidAmount) > payableTotal && payableTotal > 0) {
                toast.error("Partial payment cannot exceed payable amount")
                return
            }

            if (paidAmount > payableTotal) {
                toast.error("Paid amount cannot exceed total")
                return
            }
        }

        const payload = buildDraftPayload()

        if (payload.items.length === 0) {
            toast.error("Add at least one valid item")
            return
        }

        setCreatingInvoice(true)

        try {
            const { id: currentDraftId } = await persistDraft(payload)

            const finalizeRes = await axios.post(
                `${API}/invoices/draft/${currentDraftId}/finalize`,
                {
                    payment_status: effectivePaymentStatus,
                    paid_amount: !isAdvanceCovered && effectivePaymentStatus !== "pending" ? Number(paidAmount || 0) : 0,
                    use_advance: Boolean(useAdvance && appliedAdvance > 0)
                },
                {
                    params: {
                        payment_mode: normalizedPaymentMode
                    }
                }
            )

            toast.success("Invoice created successfully!")

            clearDraftPersistence()
            resetForm()
            setCreatedInvoice({
                invoice_number: finalizeRes.data.invoice_number || `INV-${currentDraftId}`,
                total: total
            })
            setShowSuccessModal(true)
            setShowPaymentDrawer(false)

            fetchInvoices()

        } catch (err) {
            console.error("Invoice finalize error:", err)
            toast.error(err?.response?.data?.detail || err?.message || "Failed to finalize invoice")
        } finally {
            setCreatingInvoice(false)
        }
    }

    // ============= API CALLS =============

    useEffect(() => {
        const token = localStorage.getItem("token")
        if (token) {
            axios.defaults.headers.common["Authorization"] = `Bearer ${token}`
        }
        fetchProducts()
        fetchCombos()
    }, [])
    useEffect(() => {
        if (location.state?.tab) {
            setActiveTab(location.state.tab)
        }
    }, [location.state])

    useEffect(() => {
        fetchInvoices()
    }, [page, statusFilter, rangeFilter, monthFilter])

    useEffect(() => {
        fetchTodayStats()
    }, [])

    useEffect(() => {
        initializePdfDownloadSupport().catch(() => {
            // keep downloads working even if notification permission fails
        })
    }, [])

    useEffect(() => {
        const params = new URLSearchParams(location.search)
        if (params.get("action") !== "export") {
            return
        }

        setActiveTab("list")
        setExportDialogOpen(true)
        navigate(location.pathname, {
            replace: true,
            state: { ...(location.state || {}), tab: "list" },
        })
    }, [location.pathname, location.search, location.state, navigate])

    useEffect(() => {
        if (activeTab === "create" && scanMode === "barcode") {
            setTimeout(() => skuInputRef.current?.focus(), 300)
        }
    }, [activeTab, scanMode])

    useEffect(() => {
        setScanMode("barcode")
        setShowCamera(false)
        setTimeout(() => skuInputRef.current?.focus(), 300)
    }, [])

    useEffect(() => {
        if (!showVariantDialog) return
        const el = document.getElementById(`variant-${selectedVariantIndex}`)
        if (el) {
            el.scrollIntoView({ behavior: "smooth", block: "nearest" })
        }
    }, [selectedVariantIndex, showVariantDialog])

    useEffect(() => {
        try {
            const rawState = localStorage.getItem(INVOICE_STORAGE_KEY)

            if (!rawState) {
                hasRestoredDraftRef.current = true
                return
            }

            const savedState = JSON.parse(rawState)

            const restoredPhone = normalizePhoneNumber(savedState.customerPhone || "")

            setCustomerPhone(restoredPhone)
            setCustomerName(savedState.customerName || "")
            setCustomerEmail(savedState.customerEmail || "")
            setCustomerAddress(savedState.customerAddress || "")
            setCustomerId(savedState.customerId ?? null)
            setCustomerBalance(normalizeCustomerBalance(savedState.customerBalance || 0))
            setUseAdvance(Boolean(savedState.useAdvance))
            setLineItems(Array.isArray(savedState.lineItems) ? savedState.lineItems : [])
            setDiscount(Number(savedState.discount || 0))
            setPaymentStatus(normalizePaymentStatus(savedState.paymentStatus))
            setPaymentMode(normalizePaymentMode(savedState.paymentMode))
            setGstEnabled(Boolean(savedState.gstEnabled))
            setGstRate(Number(savedState.gstRate || 18))
            setAdditionalCharges(
                Array.isArray(savedState.additionalCharges) && savedState.additionalCharges.length > 0
                    ? savedState.additionalCharges
                    : [{ label: "", amount: "" }]
            )
            setDraftId(savedState.draftId || null)
            setIsDraft(Boolean(savedState.draftId || savedState.isDraft))
            setPaidAmount(Number(savedState.paidAmount || 0))

            if (savedState.customerId && restoredPhone.length === 10) {
                setTimeout(() => {
                    searchCustomerByPhone(restoredPhone, { silent: true })
                }, 0)
            }
        } catch (error) {
            console.error("Failed to restore invoice draft state:", error)
            localStorage.removeItem(INVOICE_STORAGE_KEY)
        } finally {
            hasRestoredDraftRef.current = true
        }
    }, [])

    useEffect(() => {
        if (!hasRestoredDraftRef.current) return

        const timeout = setTimeout(() => {
            const invoiceState = {
                customerPhone, customerName, customerEmail, customerAddress, customerId,
                customerBalance, useAdvance,
                lineItems,
                discount,
                paymentStatus: normalizedPaymentStatus,
                paymentMode: normalizedPaymentMode,
                gstEnabled,
                gstRate,
                additionalCharges, draftId, isDraft, paidAmount
            }
            localStorage.setItem(INVOICE_STORAGE_KEY, JSON.stringify(invoiceState))
        }, 800)
        return () => clearTimeout(timeout)
    }, [customerPhone, customerName, customerEmail, customerAddress, customerId, customerBalance, useAdvance, lineItems, discount, normalizedPaymentStatus, normalizedPaymentMode, gstEnabled, gstRate, additionalCharges, draftId, isDraft, paidAmount])

    useEffect(() => {
        if (location.state?.customer) {
            const c = location.state.customer
            setCustomerId(c.id)
            setCustomerName(c.name || "")
            setCustomerPhone(normalizePhoneNumber(c.phone || ""))
            setCustomerEmail(c.email || "")
            setCustomerAddress(c.address || "")
            setCustomerBalance(normalizeCustomerBalance(c.current_balance || 0))
            setUseAdvance(false)
            setActiveTab("create")
            toast.success("Customer details loaded")
            window.history.replaceState({}, document.title)
        }
    }, [location.state])

    useEffect(() => {
        draftIdRef.current = draftId
    }, [draftId])

    useEffect(() => {
        if (pendingBalance > 0 && useAdvance) {
            setUseAdvance(false)
            return
        }

        if (advanceAvailable <= 0 && useAdvance) {
            setUseAdvance(false)
            return
        }

        if (!isAdvanceCovered && useAdvance && appliedAdvance > 0 && normalizedPaymentStatus !== "paid") {
            setPaymentStatus("partial")
            return
        }

        if (isAdvanceCovered) {
            setPaidAmount(0)
        } else if (normalizedPaymentStatus === "paid") {
            setPaidAmount(payableTotal)
        } else if (normalizedPaymentStatus === "pending") {
            setPaidAmount(0)
        } else if (normalizedPaymentStatus === "partial") {
            if (paidAmount > payableTotal) {
                setPaidAmount(payableTotal)
            } else if (paidAmount < 0) {
                setPaidAmount(0)
            }
        }
    }, [normalizedPaymentStatus, payableTotal, isAdvanceCovered, useAdvance, appliedAdvance, pendingBalance, advanceAvailable, paidAmount])

    useEffect(() => {
        if (!isDraft && !draftId) return
        const timer = setTimeout(() => {
            autoSaveDraft()
        }, 1200)
        return () => clearTimeout(timer)
    }, [
        customerId,
        customerPhone,
        customerEmail,
        customerAddress,
        useAdvance,
        lineItems,
        additionalCharges,
        discount,
        gstEnabled,
        gstRate,
        customerName,
        normalizedPaymentStatus,
        normalizedPaymentMode,
        paidAmount,
    ])

    useEffect(() => {
        if (showVariantDialog) {
            setSelectedVariantIndex(0)
        }
    }, [showVariantDialog])

    const handleVariantKeyDown = (e) => {
        const variants = selectedProductForVariant?.variants || []
        if (!variants.length) return

        if (e.key === "ArrowDown") {
            e.preventDefault()
            setSelectedVariantIndex((prev) => prev < variants.length - 1 ? prev + 1 : prev)
        }
        if (e.key === "ArrowUp") {
            e.preventDefault()
            setSelectedVariantIndex((prev) => prev > 0 ? prev - 1 : prev)
        }
        if (e.key === "Enter") {
            e.preventDefault()
            const variant = variants[selectedVariantIndex]
            if (variant) {
                addSelectedProductToInvoice(selectedProductForVariant, variant)
            }
        }
    }

    useEffect(() => {
        setSelectedSearchIndex(-1)
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

    useEffect(() => {
        if (selectedSearchIndex >= 0 && resultRefs.current[selectedSearchIndex]) {
            resultRefs.current[selectedSearchIndex].scrollIntoView({
                behavior: "smooth",
                block: "nearest"
            })
        }
    }, [selectedSearchIndex])

    const handleProductSearchKeyDown = (e) => {
        if (!productSearchResults.length) return

        if (e.key === "ArrowDown") {
            e.preventDefault()
            setSelectedSearchIndex((prev) => prev < productSearchResults.length - 1 ? prev + 1 : prev)
        } else if (e.key === "ArrowUp") {
            e.preventDefault()
            setSelectedSearchIndex((prev) => (prev > 0 ? prev - 1 : -1))
        } else if (e.key === "Enter") {
            e.preventDefault()
            if (selectedSearchIndex >= 0) {
                selectProduct(productSearchResults[selectedSearchIndex])
                setSelectedSearchIndex(-1)
            }
        } else if (e.key === "Escape") {
            setProductSearchResults([])
            setSelectedSearchIndex(-1)
        }
    }

    const handleDeleteDraft = async (idToDelete) => {
        if (!idToDelete) return
        const confirmDelete = window.confirm("Are you sure you want to delete this draft? This cannot be undone.")
        if (!confirmDelete) return

        try {
            await axios.delete(`${API}/invoices/draft/${idToDelete}`)
            toast.success("Draft deleted")
            fetchDrafts()
            if (draftId === idToDelete) {
                resetForm()
            }
        } catch (err) {
            console.error("Delete draft failed", err)
            toast.error(err?.response?.data?.detail || "Failed to delete draft")
        }
    }

    const fetchInvoices = async () => {
        try {
            setLoading(true)
            const isTodayFilter = rangeFilter === "today"
            const res = await axios.get(`${API}/invoices`, {
                params: {
                    page: isTodayFilter ? 1 : page,
                    limit: isTodayFilter ? 1000 : limit,
                    status: statusFilter === "all" ? undefined : statusFilter,
                    range: rangeFilter === "all" ? undefined : rangeFilter,
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
                    pdf_url: getInvoicePdfPath(inv),
                }
            })

            setInvoices(parsedInvoices)
            setPagination(
                isTodayFilter
                    ? { page: 1, total_pages: 1 }
                    : (res.data.pagination || { page: 1, total_pages: 1 })
            )
        } catch (err) {
            console.error("Invoice fetch failed", err)
            toast.error(err?.response?.data?.detail || "Failed to load invoices")
        } finally {
            setLoading(false)
        }
    }

    const fetchTodayStats = async () => {
        try {
            const res = await axios.get(`${API}/dashboard/today`)
            setTodayStats({
                total_sales_today: Number(res.data?.total_sales_today || 0),
                invoices_today: Number(res.data?.invoices_today || 0),
                cash_sales_today: Number(res.data?.cash_sales_today || 0),
                online_sales_today: Number(res.data?.online_sales_today || 0),
            })
        } catch {
            // keep compact invoice screen usable even if stats fail
        }
    }

    const exportInvoices = async () => {
        try {
            setExportingInvoices(true)
            const token = localStorage.getItem("token")
            const response = await fetch(
                `${API}/invoices/export/pdf?range=${encodeURIComponent(exportRange)}&status=${encodeURIComponent(statusFilter)}`,
                {
                    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
                }
            )

            if (!response.ok) {
                throw new Error("Export failed")
            }

            const blob = await response.blob()
            const downloadResult = await saveFileWithNotification({
                blob,
                fileName: `invoices-${exportRange}.pdf`,
                notificationTitle: "Invoice Downloaded 📄",
                notificationBody: `invoices-${exportRange}.pdf saved to Files`,
            })
            setExportDialogOpen(false)
            toast.success(
                isNativePlatform()
                    ? (
                        downloadResult.notificationsGranted
                            ? "Invoice export saved. Check your notification."
                            : "Invoice export saved to Files. Notification permission denied."
                    )
                    : "Invoice export downloaded"
            )
        } catch {
            toast.error("Failed to export invoices")
        } finally {
            setExportingInvoices(false)
        }
    }

    const addPayment = async (invoice) => {
        setViewInvoice(invoice)
        setPaymentAmount("")
        setPaymentModeSelected("upi")
        setPaymentReference("")
        setIsPaymentModalOpen(true)
        await fetchPaymentHistory(invoice.id)
    }

    const fetchPaymentHistory = async (invoiceId) => {
        try {
            setLoadingPayments(true)
            const res = await axios.get(`${API}/invoices/${invoiceId}/payments`)
            setPaymentHistory(res.data || [])
        } catch (err) {
            console.error("Payment history fetch failed", err)
            setPaymentHistory([])
        } finally {
            setLoadingPayments(false)
        }
    }

    const handleAddPayment = async () => {
        if (!paymentAmount || Number(paymentAmount) <= 0) {
            toast.error("Enter a valid payment amount")
            return
        }
        if (!paymentModeSelected) {
            toast.error("Select a payment mode")
            return
        }

        try {
            const res = await axios.post(
                `${API}/invoices/${viewInvoice.id}/add-payment`,
                {
                    amount: Number(paymentAmount),
                    payment_mode: paymentModeSelected,
                    reference: paymentReference || null
                }
            )

            toast.success("Payment recorded successfully")
            setPaymentAmount("")
            setPaymentReference("")
            setPaymentModeSelected("upi")
            await fetchPaymentHistory(viewInvoice.id)
            fetchInvoices()

            const updatedInvoice = {
                ...viewInvoice,
                paid_amount: res.data.paid_amount,
                balance_amount: res.data.balance_amount,
                payment_status: res.data.payment_status
            }
            setViewInvoice(updatedInvoice)
        } catch (err) {
            toast.error(err?.response?.data?.detail || "Failed to add payment")
        }
    }

    const handleDeletePayment = async (paymentId) => {
        toast.error("Recorded payments cannot be deleted because the audit trail is locked")
    }

    const handleMarkAsPaid = (invoiceId) => {
        setPendingPaidInvoiceId(invoiceId)
        setPaidPaymentMode("upi")
        setPaidPaymentRef("")
        setIsPaidModeDialogOpen(true)
    }

    const confirmMarkAsPaid = async () => {
        if (!pendingPaidInvoiceId) return

        try {
            await axios.patch(
                `${API}/invoices/${pendingPaidInvoiceId}/status`,
                null,
                {
                    params: {
                        payment_status: "paid",
                        payment_mode: paidPaymentMode,
                        reference: paidPaymentRef || null
                    }
                }
            )

            toast.success("Invoice marked as paid")
            fetchInvoices()
            setIsPaidModeDialogOpen(false)
            setPendingPaidInvoiceId(null)

            setViewInvoice((prev) =>
                prev?.id === pendingPaidInvoiceId
                    ? {
                        ...prev,
                        payment_status: "paid",
                        payment_mode: paidPaymentMode,
                        paid_amount: Number(prev.paid_amount || 0) + Number(prev.balance_amount || 0),
                        balance_amount: 0
                    }
                    : prev
            )
        } catch (err) {
            toast.error(err?.response?.data?.detail || "Failed to mark as paid")
        }
    }

    const fetchProducts = async () => {
        try {
            const res = await axios.get(`${API}/products`, {
                params: { limit: 100 }
            })
            setProducts(Array.isArray(res.data) ? res.data : res.data?.data || [])
        } catch (err) {
            console.error("Product fetch failed", err)
            toast.error("Failed to load products")
        }
    }

    const fetchCombos = async () => {
        try {
            const res = await axios.get(`${API}/combos`)
            setCombos(res.data || [])
        } catch (err) {
            console.error("Combo fetch failed", err)
            toast.error("Failed to load combos")
        }
    }

    const cleanedAdditionalCharges = additionalCharges.filter(
        c => c.label?.trim() || Number(c.amount) > 0
    )

    const handleSaveDraft = async (silent = false) => {
        if (lineItems.length === 0) return
        if (creatingInvoice) return
        if (!customerName) {
            if (!silent) toast.error("Customer name required")
            return
        }

        try {
            const draftPayload = buildDraftPayload()

            console.log("Saving draft", { draftId, payload: draftPayload })
            const { createdNewDraft } = await persistDraft(draftPayload)

            if (!silent && !draftToastLock.current) {
                draftToastLock.current = true
                toast.success(
                    createdNewDraft ? "Draft saved successfully" : "Draft updated successfully",
                    { duration: 3000 }
                )
                setTimeout(() => {
                    draftToastLock.current = false
                }, 3000)
            }
        } catch (err) {
            console.error("Draft save failed:", err)
            if (!silent) {
                toast.error(err?.response?.data?.detail || err?.message || "Failed to save draft")
            }
        }
    }

    const autoSaveDraft = () => {
        if (!customerName) return
        if (lineItems.length === 0) return
        handleSaveDraft(true)
    }

    const fetchDrafts = async () => {
        try {
            const res = await axios.get(`${API}/invoices/drafts`)
            const list = Array.isArray(res.data) ? res.data : Array.isArray(res.data?.data) ? res.data.data : []
            setDrafts(list)
        } catch (err) {
            console.error("Draft fetch failed", err)
            toast.error("Failed to load drafts")
            setDrafts([])
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
            setSelectedProductForVariant(product)
            setShowVariantDialog(true)
        } else {
            addProductToInvoice(product)
        }
        setProductSearch("")
        setSearchResults([])
        setProductSearchQuery("")
        setProductSearchResults([])
    }

    const addComboToInvoice = (combo) => {
        if (!combo.items || combo.items.length === 0) {
            toast.error("Combo has no items")
            return
        }

        const variantItems = combo.items.filter((item) => item.has_variants)

        if (variantItems.length > 0) {
            const first = variantItems[0]

            setPendingCombo({
                combo,
                variantItems,
                selected: [],
                currentIndex: 0,
            })

            setSelectedProductForVariant({
                id: first.product_id,
                name: first.product_name,
                variants: first.variants || [],
                selling_price: Number(first.selling_price || first.price || 0),
                images: first.image_url ? [first.image_url] : [],
            })

            setShowVariantDialog(true)
            setComboModal(false)
            return
        }

        expandComboItems(combo, [])
        setComboModal(false)
    }

    const expandComboItems = (combo, selectedVariants) => {
        const newItems = combo.items.map((item) => {
            const selected = selectedVariants.find((v) => v.product_id === item.product_id)
            const variant = selected?.variant_info

            let finalPrice = 0

            if (selected?.price && Number(selected.price) > 0) {
                finalPrice = Number(selected.price)
            } else if (variant?.v_selling_price && Number(variant.v_selling_price) > 0) {
                finalPrice = Number(variant.v_selling_price)
            } else if (item?.selling_price && Number(item.selling_price) > 0) {
                finalPrice = Number(item.selling_price)
            } else if (item?.price && Number(item.price) > 0) {
                finalPrice = Number(item.price)
            }

            return {
                product_id: item.product_id,
                combo_id: combo.id,
                sku: selected?.sku || item.sku || null,
                v_sku: selected?.sku || null,
                product_name: variant
                    ? `${item.product_name} (${variant.variant_name || variant.color || variant.size || "Variant"})`
                    : item.product_name,
                quantity: item.quantity || 1,
                price: finalPrice,
                gst_rate: gstEnabled ? Number(gstRate) : 0,
                total: finalPrice * (item.quantity || 1),
                is_service: 0,
                image_url: variant?.image_url || item.image_url || "/placeholder.png",
                variant_info: variant || null,
                variant_name: variant?.variant_name || null,
                color: variant?.color || null,
                size: variant?.size || null,
                is_combo_item: true,
            }
        })

        setLineItems((prev) => [...prev, ...newItems])
        toast.success("Combo added")
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

    const searchCustomerByPhone = async (phone, options = {}) => {
        const { silent = false } = options

        if (phone.length < 10) return

        try {
            const res = await axios.get(`${API}/customers/search?phone=${phone}`)

            if (res.data) {
                setCustomerId(res.data.id)
                setCustomerName(res.data.name)
                setCustomerEmail(res.data.email)
                setCustomerAddress(res.data.address || "")
                setCustomerBalance(normalizeCustomerBalance(res.data.current_balance || 0))
                setUseAdvance(false)
                if (!silent) {
                    toast.success("Customer found!")
                }
            } else {
                setCustomerId(null)
                setCustomerName("")
                setCustomerEmail("")
                setCustomerAddress("")
                setCustomerBalance(0)
                setUseAdvance(false)
                if (!silent) {
                    toast.info("New customer - please fill details")
                }
            }
        } catch (err) {
            console.error("Customer search failed", err)
            if (!silent) {
                toast.error("Customer lookup failed")
            }
        }
    }

    const updateLineItem = (index, field, value) => {
        const updated = [...lineItems]
        const numericValue =
            field === "quantity" || field === "price"
                ? Math.max(field === "quantity" ? 1 : 0, Number(value) || 0)
                : value

        updated[index][field] = numericValue

        if (field === "product_id") {
            const product = products.find((p) => String(p.id) === String(value))
            if (product) {
                const hasVariants = product.variants && product.variants.length > 0

                if (hasVariants && product.variants.length === 1) {
                    const variant = product.variants[0]
                    updated[index].product_name =
                        `${product.name} (${variant.variant_name || variant.color || variant.size || "Variant"})`
                    updated[index].price = variant.v_selling_price || product.selling_price
                    updated[index].v_sku = variant.v_sku
                    updated[index].variant_name = variant.variant_name
                    updated[index].color = variant.color
                    updated[index].size = variant.size
                    updated[index].image_url = variant.image_url || (product.images && product.images[0]) || "/placeholder.png"
                    updated[index].variant_info = variant
                } else {
                    updated[index].product_name = product.name
                    updated[index].price = product.selling_price
                    updated[index].v_sku = null
                    updated[index].variant_name = null
                    updated[index].color = null
                    updated[index].size = null
                    updated[index].image_url = (product.images && product.images[0]) || "/placeholder.png"
                    updated[index].variant_info = null
                }

                updated[index].total =
                    Number(updated[index].price || 0) * Number(updated[index].quantity || 0)
            }
        }

        if (field === "quantity" || field === "price") {
            updated[index].total =
                Number(updated[index].price || 0) * Number(updated[index].quantity || 0)
        }

        setLineItems(updated)
    }

    const handleSkuScan = async (skuInputRaw) => {
        if (!skuInputRaw) return
        if (scanLockRef.current) return
        scanLockRef.current = true

        let scannedSku = skuInputRaw.trim()

        try {
            const parsed = JSON.parse(scannedSku)
            scannedSku = parsed.v_sku || parsed.sku || scannedSku
        } catch { }

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

            if (!isVariant && hasVariants) {
                setSelectedProductForVariant(product)
                setShowVariantDialog(true)
                setSkuInput("")
                return
            }

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
        if (activeTab === "drafts") {
            fetchDrafts()
        }
    }, [activeTab])

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
        setLineItems((prev) => prev.filter((_, i) => i !== index))
    }

    const resetForm = () => {
        setCustomerPhone("")
        setCustomerName("")
        setCustomerEmail("")
        setCustomerAddress("")
        setCustomerId(null)
        setCustomerBalance(0)
        setUseAdvance(false)
        setLineItems([])
        setDiscount(0)
        setPaymentStatus("pending")
        setPaymentMode("upi")
        setDraftId(null)
        setIsDraft(false)
        setAdditionalCharges([{ label: "", amount: "" }])
        setSkuInput("")
        setProductSearch("")
        setSearchResults([])
        setShowProductSearch(false)
        setSelectedProduct(null)
        setShowVariantDialog(false)
        setComboModal(false)
        setPendingCombo(null)
        setProductSearchQuery("")
        setProductSearchResults([])
        setIsSearchingProducts(false)
        setSelectedProductForVariant(null)
        setGstEnabled(false)
        setGstRate(18)
        setPaidAmount(0)
        localStorage.removeItem(INVOICE_STORAGE_KEY)
    }

const generatePDF = async (invoice) => {
        const pdfPath = getInvoicePdfPath(invoice)
        if (pdfPath) {
            try {
                const pdfUrl = resolveInvoicePdfUrl(pdfPath)
                const token = localStorage.getItem("token")
                const pdfRes = await fetch(pdfUrl, {
                    headers: token ? { Authorization: `Bearer ${token}` } : undefined,
                })
                if (!pdfRes.ok) {
                    throw new Error("Failed to fetch invoice PDF")
                }

                const pdfBlob = await pdfRes.blob()
                const downloadResult = await savePdfWithNotification({
                    blob: pdfBlob,
                    fileName: `${invoice.invoice_number}.pdf`,
                })
                toast.success(
                    isNativePlatform()
                        ? (
                            downloadResult.notificationsGranted
                                ? "Invoice saved. Check your notification."
                                : "Invoice saved to Files. Notification permission denied."
                        )
                        : "Invoice PDF downloaded"
                )
                return
            } catch (err) {
                console.error("Invoice download failed:", err)
                toast.error(err?.message || "Failed to download invoice PDF")
                return
            }
        }

        const printWindow = window.open("", "_blank")
        const subtotal = invoice.subtotal
// const additionalAmount = invoice.additional_amount || 0
// const additionalLabel = invoice.additional_label || "Additional Charge"
const gstAmount = invoice.gst_amount
const total = getInvoiceDisplayTotal(invoice)
        const invoiceHtml = buildInvoicePrintHtml({
          invoice,
          subtotal,
          gstAmount,
          total,
          formatCurrency,
        })


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
                <div class="company-name">R RIDE</div>
    <div class="company-tagline" style="font-weight: bold;">BIKE GARAGE & STUDIO</div>

                <div class="company-details">
                  <p>Mahaveer Building, gala no 09, near payal talkies, mandai road,
    Dhamankar naka, Bhiwandi, 421305 </p>
                  <p>Phone: +91 90822 582580 | Email: outransystems@gmail.com</p>
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
                  <p><strong>Created By:</strong> ${invoice.created_by || "—"}</p>

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
    <th class="text-right">
      ${invoice.gst_enabled ? `GST ${invoice.gst_rate}%` : "GST"}
    </th>
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
                      <td class="text-right">${formatDisplayCurrency(item.price)}</td>
                      <td class="text-right">${item.gst_rate}%</td>
                      <td class="text-right">${formatDisplayCurrency(item.total)}</td>
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
                    <td class="text-right">${formatDisplayCurrency(subtotal)}</td>
                  </tr>
                  <tr>
    ${gstAmount > 0 ? `
    <tr>
      <td>GST Amount:</td>
      <td class="text-right">${formatDisplayCurrency(gstAmount)}</td>
    </tr>
    ` : ""}
                  </tr>
              ${Array.isArray(invoice.additional_charges)
  ? invoice.additional_charges
      .filter(c => Number(c.amount) > 0)
      .map(c => `
        <tr>
          <td>${c.label || "Additional Charge"}:</td>
          <td class="text-right">${formatDisplayCurrency(c.amount)}</td>
        </tr>
      `).join("")
  : ""
}

                  <tr>
                    <td>Discount:</td>
                    <td class="text-right">-${formatDisplayCurrency(invoice.discount)}</td>
                  </tr>
                  <tr class="total-row">
                    <td>Total Amount:</td>
                    <td class="text-right">${formatDisplayCurrency(total)}</td>
                  </tr>
                  ${invoice.payment_status === "partial" ? `
                  <tr>
                    <td>Paid Amount:</td>
                    <td class="text-right">${formatDisplayCurrency(invoice.paid_amount || 0)}</td>
                  </tr>
                  <tr style="background: #fff3cd;">
                    <td><strong>Remaining Balance:</strong></td>
                    <td class="text-right" style="color: #c41e3a;"><strong>${formatDisplayCurrency(invoice.balance_amount || 0)}</strong></td>
                  </tr>
                  ` : ""}
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

        printWindow.document.write(invoiceHtml)
        printWindow.document.close()
      }
    const shareInvoice = async (invoice) => {
        if (Capacitor.isNativePlatform()) {
            try {
                const plugins = window.Capacitor?.Plugins || {}
                const Filesystem = plugins.Filesystem
                const Share = plugins.Share

                if (!Filesystem || !Share) {
                    throw new Error("Native share plugins are not available")
                }

                const pdfPath = getInvoicePdfPath(invoice)

                if (!pdfPath) {
                    throw new Error("Invoice PDF is not available yet")
                }

                setSharingInvoiceId(invoice.id)
                const pdfUrl = resolveInvoicePdfUrl(pdfPath)

                const pdfRes = await fetch(pdfUrl, {
                    headers: {
                        Authorization: `Bearer ${localStorage.getItem("token")}`,
                    },
                })

                if (!pdfRes.ok) {
                    throw new Error("Failed to fetch invoice PDF")
                }

                const pdfBlob = await pdfRes.blob()
                const base64Data = await blobToBase64(pdfBlob)
                const fileName = `${invoice.invoice_number}.pdf`

                const writeResult = await Filesystem.writeFile({
                    path: fileName,
                    data: base64Data,
                    directory: "CACHE",
                    recursive: true,
                })

                const fileUri = writeResult?.uri
                    || (await Filesystem.getUri({
                        path: fileName,
                        directory: "CACHE",
                    })).uri

                await Share.share({
                    title: "Invoice",
                    text: `Invoice ${invoice.invoice_number}`,
                    url: fileUri,
                    dialogTitle: "Share Invoice",
                })

                return
            } catch (err) {
                console.error("Native invoice share failed:", err)
                toast.error(err?.message || "Failed to share invoice PDF")
                return
            } finally {
                setSharingInvoiceId(null)
            }
        }

        try {
            const element = document.createElement("div")
            const subtotal = invoice.subtotal
            const gstAmount = invoice.gst_amount
            const total = getInvoiceDisplayTotal(invoice)

            element.innerHTML = `
<div style="padding: 20px;">
  <h2>R RIDE - BIKE GARAGE & STUDIO</h2>
  <p><strong>Invoice No:</strong> ${invoice.invoice_number}</p>
  <p><strong>Date:</strong> ${new Date(invoice.created_at).toLocaleDateString()}</p>
  <hr>
  <p><strong>Customer:</strong> ${invoice.customer_name}</p>
  <p><strong>Phone:</strong> ${invoice.customer_phone || "N/A"}</p>

  ${invoice.items.map((item, i) => `
    <table style="width: 100%; margin-top: 10px;">
      <tr>
        <th>#</th>
        <th>Product</th>
        <th>Qty</th>
        <th>Price</th>
        <th>Total</th>
      </tr>
      <tr>
        <td>${i + 1}</td>
        <td>${item.product_name}</td>
        <td>${item.quantity}</td>
        <td>${formatDisplayCurrency(item.price)}</td>
        <td>${formatDisplayCurrency(item.total)}</td>
      </tr>
    </table>
  `).join("")}

  <p><strong>Subtotal:</strong> ${formatDisplayCurrency(subtotal)}</p>
  <p><strong>GST:</strong> ${formatDisplayCurrency(gstAmount)}</p>
  <p><strong>Discount:</strong> ${formatDisplayCurrency(invoice.discount)}</p>
  <h3>Total: ${formatDisplayCurrency(total)}</h3>
  ${invoice.payment_status === "partial" ? `
    <div style="background: #fff3cd; padding: 10px; border-radius: 5px; margin-top: 10px;">
      <p><strong>Paid Amount:</strong> ${formatDisplayCurrency(invoice.paid_amount || 0)}</p>
      <p style="color: #c41e3a;"><strong>Remaining Balance:</strong> ${formatDisplayCurrency(invoice.balance_amount || 0)}</p>
    </div>
  ` : ""}
</div>
      `

            const pdfBlob = await html2pdf().from(element).outputPdf("blob")

            const file = new File(
                [pdfBlob],
                `Invoice-${invoice.invoice_number}.pdf`,
                { type: "application/pdf" }
            )

            if (navigator.canShare && navigator.canShare({ files: [file] })) {
                await navigator.share({
                    title: `Invoice ${invoice.invoice_number}`,
                    files: [file],
                })
            } else {
                alert("File sharing not supported on this device")
            }

        } catch (err) {
            console.error("Share failed:", err)
        }
    }

    const updateInvoiceStatus = async (invoiceId, newStatus) => {
        const targetInvoice =
            invoices.find((invoice) => invoice.id === invoiceId) ||
            (viewInvoice?.id === invoiceId ? viewInvoice : null)

        if (newStatus === "paid") {
            handleMarkAsPaid(invoiceId)
            return
        }

        if (newStatus === "partial") {
            if (targetInvoice) {
                addPayment(targetInvoice)
            } else {
                toast.info("Use Add Payment to record a partial payment")
            }
            return
        }

        if (newStatus === "pending") {
            toast.info("Pending is set automatically. Use payment or cancel actions instead.")
            return
        }

        if (newStatus === "cancelled" && targetInvoice?.payment_status !== "pending") {
            toast.error("Only pending invoices can be cancelled")
            return
        }

        try {
            await axios.patch(`${API}/invoices/${invoiceId}/status`, null, {
                params: { payment_status: newStatus },
            })

            toast.success("Invoice status updated")
            if (viewInvoice?.id === invoiceId) {
                setViewInvoice((prev) => (prev ? { ...prev, payment_status: newStatus } : prev))
            }
            fetchInvoices()
        } catch (err) {
            toast.error(err?.response?.data?.detail || "Failed to update status")
        }
    }

    const getInvoiceTotal = (inv) => {
        const itemsTotal = Array.isArray(inv.items)
            ? inv.items.reduce((s, i) => s + Number(i.total || 0), 0)
            : 0

        const extras = Array.isArray(inv.additional_charges)
            ? inv.additional_charges.reduce(
                (s, c) => s + Number(c.amount || 0),
                0
            )
            : 0

        return (
            itemsTotal +
            extras +
            Number(inv.gst_amount || 0) -
            Number(inv.discount || 0)
        )
    }

    const filteredInvoices = invoices.filter((inv) => invoiceMatchesSearch(inv, searchTerm))

    const mobileStatsCards = [
        {
            label: "Today Sales",
            value: `${formatDisplayCurrency(todayStats.total_sales_today)}`,
            icon: Wallet,
        },
        {
            label: "Today Invoices",
            value: formatNumber(todayStats.invoices_today),
            icon: Receipt,
        },
        {
            label: "Cash",
            value: `${formatDisplayCurrency(todayStats.cash_sales_today)}`,
            icon: Landmark,
        },
        {
            label: "Online",
            value: `${formatDisplayCurrency(todayStats.online_sales_today)}`,
            icon: Share2,
        },
    ]

    const loadDraft = (draft) => {
        if (!draft || !draft.id) {
            toast.error("Invalid draft")
            return
        }

        setDraftId(draft.id)
        setIsDraft(true)
        setCustomerId(draft.customer_id ?? null)
        setCustomerName(draft.customer_name ?? "")
        setCustomerPhone(normalizePhoneNumber(draft.customer_phone ?? ""))
        setCustomerAddress(draft.customer_address ?? "")
        setCustomerEmail("")

        const parsedItems = Array.isArray(draft.items) ? draft.items : []

        setLineItems(
            parsedItems.map((item) => ({
                product_id: item.product_id,
                combo_id: item.combo_id || null,
                product_name: item.product_name,
                quantity: Number(item.quantity || 1),
                price: Number(item.price || 0),
                total: Number(item.total || 0),
                gst_rate: Number(item.gst_rate || 0),
                sku: item.sku || null,
                v_sku: item.v_sku || item.sku || null,
                is_service: item.is_service || 0,
                image_url: item.image_url || "/placeholder.png",
                variant_info: item.variant_info || null,
            }))
        )

        setDiscount(Number(draft.discount || 0))
        setPaymentStatus(normalizePaymentStatus(draft.payment_status))
        setPaymentMode(normalizePaymentMode(draft.payment_mode))
        setPaidAmount(Number(draft.paid_amount || 0))

        if (Array.isArray(draft.additional_charges) && draft.additional_charges.length > 0) {
            setAdditionalCharges(
                draft.additional_charges.map((c) => ({
                    label: c.label || "",
                    amount: Number(c.amount || 0),
                }))
            )
        } else {
            setAdditionalCharges([{ label: "", amount: "" }])
        }

        setGstEnabled(Boolean(draft.gst_enabled))
        setGstRate(Number(draft.gst_rate || 18))

        setActiveTab("create")
        toast.success("Draft loaded")
    }

    const handleSelectProduct = (product) => {
        const variants = product.variants || []

        if (variants.length > 1) {
            setSelectedProductForVariant(product)
            setShowVariantDialog(true)
        } else if (variants.length === 1) {
            addSelectedProductToInvoice(product, variants[0])
        } else {
            addSelectedProductToInvoice(product, null)
        }

        setProductSearchQuery("")
        setProductSearchResults([])
    }

    const addSelectedProductToInvoice = (product, variant) => {
        if (pendingCombo) {
            const { combo, variantItems, selected, currentIndex } = pendingCombo
            const currentComboItem = variantItems[currentIndex]

            let comboPrice = 0

            if (variant?.v_selling_price && Number(variant.v_selling_price) > 0) {
                comboPrice = Number(variant.v_selling_price)
            } else if (product?.selling_price && Number(product.selling_price) > 0) {
                comboPrice = Number(product.selling_price)
            } else if (currentComboItem?.selling_price && Number(currentComboItem.selling_price) > 0) {
                comboPrice = Number(currentComboItem.selling_price)
            } else if (currentComboItem?.price && Number(currentComboItem.price) > 0) {
                comboPrice = Number(currentComboItem.price)
            }

            const newSelected = [
                ...selected,
                {
                    product_id: product.id,
                    sku: variant?.v_sku || product.sku,
                    price: comboPrice,
                    variant_info: variant || null,
                },
            ]

            if (currentIndex + 1 < variantItems.length) {
                const next = variantItems[currentIndex + 1]

                setPendingCombo({
                    combo,
                    variantItems,
                    selected: newSelected,
                    currentIndex: currentIndex + 1,
                })

                setSelectedProductForVariant({
                    id: next.product_id,
                    name: next.product_name,
                    variants: next.variants || [],
                    selling_price: Number(next.selling_price || next.price || 0),
                    images: next.image_url ? [next.image_url] : [],
                })

                return
            }

            expandComboItems(combo, newSelected)
            setPendingCombo(null)
            setShowVariantDialog(false)
            setSelectedProductForVariant(null)
            return
        }

        let displayName, itemPrice, itemImage, itemSku, variantInfo

        if (variant) {
            const variantLabel = variant.variant_name || variant.color || variant.size || "Variant"
            displayName = `${product.name} (${variantLabel})`
            itemPrice =
                (variant.v_selling_price !== null && variant.v_selling_price !== undefined
                    ? Number(variant.v_selling_price)
                    : Number(product.selling_price))
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

    // ============= RENDER =============

    return (
<div
  className="bg-background pb-32"
  style={{ minHeight: "100dvh" }}
>            {/* COMPACT TOP NAVIGATION - 48px height */}


            {/* MAIN CONTENT */}
<div
  className="px-3 pb-28 space-y-4"
  style={{ paddingTop: "calc(env(safe-area-inset-top) + 12px)" }}
>                        {/* ==================== CREATE TAB ==================== */}
                {activeTab === "create" && (
                    <div className="max-w-2xl mx-auto space-y-3">
                        {draftId && (
                            <div className="bg-blue-50 border border-blue-200 rounded-lg px-3 py-2 text-xs text-blue-900">
                                Editing Draft • #{draftId}
                            </div>
                        )}

<Card className="rounded-2xl border border-border/50 shadow-sm bg-card">
  <div className="p-3 space-y-3">

    {/* Header */}
    <div className="flex items-center justify-between">
      <h3 className="text-sm font-semibold">Customer</h3>
    </div>

    {/* Inputs */}
    <div className="grid grid-cols-2 gap-3">
      <Input
        type="number"
        inputMode="numeric"
        autoComplete="tel"
        enterKeyHint="next"
        pattern="[0-9]*"
        min="0"
        step="1"
        placeholder="Phone*"
        value={customerPhone}
        onChange={(e) => {
          const val = normalizePhoneNumber(e.target.value)
          setCustomerPhone(val)
          if (val.length === 10) {
            searchCustomerByPhone(val)
          } else {
            setCustomerId(null)
            setCustomerBalance(0)
              setUseAdvance(false)
          }
        }}
        onInput={(e) => {
          e.currentTarget.value = normalizePhoneNumber(e.currentTarget.value)
        }}
        className="h-11 text-sm rounded-xl [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />

      <Input
        type="text"
        inputMode="text"
        autoComplete="name"
        autoCapitalize="words"
        placeholder="Name*"
        value={customerName}
        onChange={(e) => setCustomerName(e.target.value)}
        className="h-11 text-sm rounded-xl"
      />
    </div>

    <div className={`mt-3 rounded-xl border p-3 ${
      pendingBalance > 0
        ? "border-orange-500/30 bg-orange-500/10"
        : advanceAvailable > 0
          ? "border-emerald-500/30 bg-emerald-500/10"
          : "border-border/60 bg-muted/20"
    }`}>
      <p className="text-sm font-semibold">{getCustomerBalanceText(customerBalance)}</p>
      {pendingBalance > 0 && (
        <p className="mt-1 text-[11px] text-orange-200">
          Advance can be used only when pending balance is zero.
        </p>
      )}
    </div>

    {advanceAvailable > 0 && (
      <div className="mt-3 rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3 space-y-2">
        <div className="flex items-center justify-between gap-2">
          <div>
                              <p className="text-sm font-semibold text-emerald-300">Advance Available: {formatDisplayCurrency(advanceAvailable)}</p>
            <p className="text-[11px] text-emerald-100/80">Use wallet balance on this invoice</p>
          </div>
          <label className="flex items-center gap-2 text-xs text-white">
            <input
              type="checkbox"
              checked={useAdvance}
              onChange={(e) => setUseAdvance(e.target.checked)}
              disabled={pendingBalance > 0}
              className="h-4 w-4 accent-emerald-500"
            />
            Use
          </label>
        </div>
        {useAdvance && (
          <div className="flex justify-between text-xs text-emerald-100">
                                <span>Advance Used: {formatDisplayCurrency(appliedAdvance)}</span>
                                <span>Payable: {formatDisplayCurrency(payableTotal)}</span>
          </div>
        )}
        {useAdvance && !isAdvanceCovered && normalizedPaymentStatus === "partial" && remainingBalance > 0 && (
          <div className="flex justify-between text-xs text-yellow-200">
            <span>Balance Amount:</span>
                                <span>{formatDisplayCurrency(remainingBalance)}</span>
          </div>
        )}
      </div>
    )}

  </div>
</Card>

                        {/* PRODUCT SEARCH */}
                        <div className="relative">
                            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                            <Input
                                placeholder="Search product by name or SKU..."
                                value={productSearchQuery}
                                onChange={(e) => setProductSearchQuery(e.target.value)}
                                onKeyDown={handleProductSearchKeyDown}
                                className="h-10 rounded-xl pl-10 pr-20 text-sm"
                                data-testid="product-search-input"
                            />
                            <div className="absolute right-2 top-1/2 flex -translate-y-1/2 items-center gap-1">
                                {productSearchQuery && (
                                    <button
                                        onClick={() => setProductSearchQuery("")}
                                        className="rounded-full p-1 hover:bg-muted"
                                    >
                                        <X className="h-4 w-4" />
                                    </button>
                                )}
                                <button
                                    type="button"
                                    onClick={() => {
                                        if (showCamera) {
                                            setShowCamera(false)
                                            setScanMode("barcode")
                                            return
                                        }

                                        setScanMode("camera")
                                        setShowCamera(true)
                                    }}
                                    className={`rounded-full p-2 transition ${
                                        showCamera
                                            ? "bg-primary text-primary-foreground"
                                            : "bg-muted text-muted-foreground hover:bg-muted/80"
                                    }`}
                                    aria-label={showCamera ? "Close QR scanner" : "Open QR scanner"}
                                >
                                    {cameraLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Camera className="h-4 w-4" />}
                                </button>
                            </div>
                        </div>

                        {showCamera && (
                            <Card className="rounded-2xl border border-border/50 bg-card/90 p-3 shadow-sm">
                                <div className="mb-3 flex items-start justify-between gap-3">
                                    <div>
                                        <p className="text-sm font-semibold">Scan QR / Barcode</p>
                                        <p className="text-xs text-muted-foreground">Point the camera at a product code to add it quickly.</p>
                                    </div>
                                    <Button
                                        type="button"
                                        variant="ghost"
                                        size="sm"
                                        className="h-8 rounded-xl px-3 text-xs"
                                        onClick={() => {
                                            setShowCamera(false)
                                            setScanMode("barcode")
                                        }}
                                    >
                                        Close
                                    </Button>
                                </div>
                                <div
                                    id="reader"
                                    className="overflow-hidden rounded-2xl bg-black [&_video]:rounded-2xl"
                                />
                            </Card>
                        )}

                        {/* Search Results */}
                        {productSearchResults.length > 0 && (
  <Card className="absolute left-0 right-0 z-40 w-full max-w-full max-h-80 overflow-y-auto overflow-x-hidden">
    <div className="p-2 space-y-1">
      {productSearchResults.map((product, index) => (
        <button
          key={product.id}
          ref={(el) => (resultRefs.current[index] = el)}
          onClick={() => selectProduct(product)}
          className={`w-full max-w-full flex items-center gap-2 p-2 rounded-lg transition-colors text-left overflow-hidden ${
            selectedSearchIndex === index
              ? "bg-primary text-primary-foreground"
              : "hover:bg-muted"
          }`}
          data-testid={`product-result-${index}`}
        >
          {/* IMAGE */}
          <img
            src={product.images?.[0] || "/placeholder.png"}
            alt={product.name}
            className="w-10 h-10 object-cover rounded flex-shrink-0"
          />

          {/* TEXT */}
          <div className="flex-1 min-w-0 max-w-full overflow-hidden">
            <p className="text-sm font-medium truncate w-full block">
              {product.name}
            </p>
            <p className="text-xs text-muted-foreground truncate w-full block">
              {product.sku} • ₹{product.selling_price}
            </p>
          </div>

          {/* VARIANT BADGE */}
          {product.variants?.length > 0 && (
            <span className="text-xs bg-muted px-2 py-1 rounded flex-shrink-0">
              {product.variants.length}v
            </span>
          )}
        </button>
      ))}
    </div>
  </Card>
                        )}

                        <div className="grid grid-cols-2 gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setComboModal(true)}
                                className="h-9 text-xs"
                                data-testid="add-combo-item-btn"
                            >
                                <Plus className="h-3 w-3 mr-1" />
                                Add Combo
                            </Button>
                            <Button
                                variant="outline"
                                size="sm"
                                onClick={() => setShowManualDialog(true)}
                                className="h-9 text-xs"
                                data-testid="add-manual-item-btn"
                            >
                                <Plus className="h-3 w-3 mr-1" />
                                Add Manual
                            </Button>
                        </div>

                        {/* LINE ITEMS - Compact Cards */}
                        <div className="space-y-2">
                            <h3 className="text-sm font-semibold px-1">Items ({lineItems.length})</h3>
                            {lineItems.length === 0 ? (
                                <div className="text-center py-8 text-sm text-muted-foreground">
                                    Add products to start your invoice
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {lineItems.map((item, index) => (
                                     <Card key={index} className="p-3 rounded-xl shadow-sm border border-border/50" data-testid={`line-item-${index}`}>
  <div className="flex items-center gap-3 w-full overflow-hidden">

    {/* IMAGE */}
   <img
  src={item.image_url || "/placeholder.png"}
  alt={item.product_name}
  onClick={() => setPreviewImage(item.image_url || "/placeholder.png")}
  className="w-12 h-12 object-cover rounded flex-shrink-0 cursor-pointer"
/>

    {/* TEXT */}
    <div className="flex-1 min-w-0 overflow-hidden">
      <p className="text-sm font-medium truncate">
        {item.product_name}
      </p>

      <p className="text-xs text-muted-foreground truncate">
        {[item.v_sku, item.color, item.size].filter(Boolean).join(" • ")}
      </p>
    </div>

    {/* RIGHT SECTION */}
    <div className="flex flex-col items-end gap-1 flex-shrink-0">

      {/* PRICE */}
      <span className="text-base font-bold whitespace-nowrap">
        {formatDisplayCurrency(item.total)}
      </span>

      {/* STEPPER */}
      <div className="flex items-center border rounded overflow-hidden">
        <button
          onClick={() =>
            updateLineItem(index, "quantity", Math.max(1, item.quantity - 1))
          }
          className="px-2 py-1 hover:bg-muted"
        >
          <Minus className="h-3 w-3" />
        </button>

        <span className="px-2 text-sm w-6 text-center">
          {item.quantity}
        </span>

        <button
          onClick={() =>
            updateLineItem(index, "quantity", item.quantity + 1)
          }
          className="px-2 py-1 hover:bg-muted"
        >
          <Plus className="h-3 w-3" />
        </button>
      </div>

      {/* DELETE */}
      <button
        onClick={() => removeLineItem(index)}
        className="text-destructive hover:bg-destructive/10 p-1 rounded"
      >
        <Trash2 className="h-3 w-3" />
      </button>
    </div>

  </div>
</Card>
                                    ))}
                                </div>
                            )}
                        </div>
                    </div>
                )}

                {/* ==================== DRAFTS TAB ==================== */}
                {activeTab === "drafts" && (
                    <div className="max-w-2xl mx-auto space-y-2">
                        <h3 className="text-lg font-semibold px-1">Draft Invoices</h3>
                        {drafts.length === 0 ? (
                            <div className="text-center py-12 text-muted-foreground">
                                No drafts found
                            </div>
                        ) : (
                            <div className="space-y-2">
                                {drafts.map((d) => (
                                    <Card key={d.id} className="p-3" data-testid={`draft-${d.id}`}>
                                        <div className="flex items-center justify-between">
                                            <button
                                                onClick={async () => {
                                                    try {
                                                        const res = await axios.get(`${API}/invoices/draft/${d.id}`)
                                                        loadDraft(res.data)
                                                    } catch (err) {
                                                        toast.error("Failed to load draft")
                                                    }
                                                }}
                                                className="flex-1 text-left"
                                            >
                                                <p className="text-sm font-semibold">{d.draft_number}</p>
                                                <p className="text-xs text-muted-foreground">
                                                    {d.customer_name} • {formatDisplayCurrency(d.total || 0)}
                                                </p>
                                            </button>
                                            <div className="flex gap-1">
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={async (e) => {
                                                        e.stopPropagation()
                                                        try {
                                                            const res = await axios.get(`${API}/invoices/draft/${d.id}`)
                                                            loadDraft(res.data)
                                                        } catch (err) {
                                                            toast.error("Failed to load draft")
                                                        }
                                                    }}
                                                    className="h-8 px-2 text-xs"
                                                    data-testid={`open-draft-${d.id}`}
                                                >
                                                    Open
                                                </Button>
                                                <Button
                                                    variant="ghost"
                                                    size="sm"
                                                    onClick={(e) => {
                                                        e.stopPropagation()
                                                        handleDeleteDraft(d.id)
                                                    }}
                                                    className="h-8 px-2 text-xs text-destructive hover:bg-destructive/10"
                                                    data-testid={`delete-draft-${d.id}`}
                                                >
                                                    <Trash2 className="h-3 w-3" />
                                                </Button>
                                            </div>
                                        </div>
                                    </Card>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* ==================== INVOICES TAB ==================== */}
                {activeTab === "list" && (
                    <div className="max-w-4xl mx-auto space-y-3">
                        <div className="grid grid-cols-4 gap-2">
                            {mobileStatsCards.map((card) => {
                                const Icon = card.icon
                                return (
                                    <Card key={card.label} className="min-w-0 overflow-hidden border border-border/60 rounded-xl shadow-sm bg-card/90">
                                        <div className="min-w-0 p-2.5">
                                            <div className="min-w-0">
                                                <p className="text-[9px] font-semibold uppercase tracking-[0.08em] text-muted-foreground leading-tight truncate">
                                                    {card.label}
                                                </p>
                                                <p className="mt-0.5 text-[11px] font-bold text-foreground leading-tight truncate">
                                                    {card.value}
                                                </p>
                                            </div>
                                        </div>
                                    </Card>
                                )
                            })}
                        </div>

                        {/* Search & Filters - Compact */}
                        <div className="space-y-2">
                            <div className="flex items-center gap-2">
                                <div className="relative flex-1">
                                    <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                                    <Input
                                        placeholder="Search invoice, customer, item, or SKU..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                        className="pl-8 pr-10 h-9 text-sm"
                                        data-testid="invoice-search-input"
                                    />
                                </div>
                                <Button
                                    type="button"
                                    variant="outline"
                                    onClick={() => setShowMobileFilters((current) => !current)}
                                    className="h-9 shrink-0 px-3 text-xs md:hidden"
                                >
                                    {showMobileFilters ? "Close" : "Filter"}
                                </Button>
                            </div>
                            <div className={`${showMobileFilters ? "flex" : "hidden"} md:flex flex-col sm:flex-row gap-2`}>
                                <Select value={statusFilter} onValueChange={(v) => { setPage(1); setStatusFilter(v) }}>
                                    <SelectTrigger className="w-full sm:w-32 h-9 text-xs">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="all">All Status</SelectItem>
                                        <SelectItem value="paid">✓ Paid</SelectItem>
                                        <SelectItem value="pending">⏳ Pending</SelectItem>
                                        <SelectItem value="partial">⚠ Partial</SelectItem>
                                        <SelectItem value="cancelled">✕ Cancelled</SelectItem>
                                    </SelectContent>
                                </Select>
                                <Select value={rangeFilter} onValueChange={(v) => { setPage(1); setRangeFilter(v) }}>
                                    <SelectTrigger className="w-full sm:w-32 h-9 text-xs">
                                        <SelectValue />
                                    </SelectTrigger>
                                    <SelectContent>
                                        <SelectItem value="today">Today</SelectItem>
                                        <SelectItem value="week">This Week</SelectItem>
                                        <SelectItem value="last10">Last 10 Days</SelectItem>
                                        <SelectItem value="last30">Last 30 Days</SelectItem>
                                        <SelectItem value="lastMonth">Last Month</SelectItem>
                                    </SelectContent>
                                </Select>
                            </div>
                        </div>

                        <div className="space-y-2">
                            <h3 className="text-sm font-semibold px-1">
                                {rangeFilter === "today" ? "Today's Invoices" : "All Invoices"} ({filteredInvoices.length})
                            </h3>
                            {filteredInvoices.length === 0 ? (
                                <div className="text-center py-12 text-muted-foreground">
                                    No invoices found
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {filteredInvoices.filter((inv) => inv && inv.id).map((inv) => (
                                        <Card
                                            key={inv.id}
                                            className="p-2.5 border border-border/50 shadow-sm rounded-xl"
                                            data-testid={`invoice-table-${inv.id}`}
                                        >
                                            <div className="space-y-2">
                                                <div className="flex items-start justify-between gap-2">
                                                    <div className="min-w-0">
                                                        <p className="text-[13px] font-semibold leading-4 truncate">{inv.invoice_number}</p>
                                                        <div className="flex items-center gap-1.5 mt-0.5 text-[11px] text-muted-foreground">
                                                            <span className="truncate max-w-[110px]">{inv.customer_name}</span>
                                                            <span>•</span>
                                                            <span className="whitespace-nowrap">{new Date(inv.created_at).toLocaleDateString()}</span>
                                                        </div>
                                                    </div>
                                                    <div className="text-right shrink-0">
                                                        <p className="text-base font-bold">{formatDisplayCurrency(getInvoiceDisplayTotal(inv))}</p>
                                                        {Number(inv.advance_used || 0) > 0 && (
                                                            <p className="mt-1 text-[10px] font-medium text-emerald-600">
                                                                Advance Used: {formatDisplayCurrency(inv.advance_used)}
                                                            </p>
                                                        )}
                                                        {inv.payment_status === "partial" && Number(inv.balance_amount || 0) > 0 && (
                                                            <p className="mt-1 text-[10px] font-medium text-red-600">
                                                                Balance: {formatDisplayCurrency(inv.balance_amount)}
                                                            </p>
                                                        )}
                                                        <span className={`inline-flex mt-1 px-1.5 py-0.5 rounded text-[9px] font-medium ${inv.payment_status === "paid" ? "bg-green-100 text-green-800" :
                                                                inv.payment_status === "pending" ? "bg-yellow-100 text-yellow-800" :
                                                                    inv.payment_status === "partial" ? "bg-blue-100 text-blue-800" :
                                                                        "bg-gray-100 text-gray-800"
                                                            }`}>
                                                            {inv.payment_status.toUpperCase()}
                                                        </span>
                                                    </div>
                                                </div>

                                                <div className="flex items-center justify-between gap-2">
                                                    <p className="text-[11px] text-muted-foreground truncate min-w-0">
                                                        {inv.customer_phone || "No phone"}
                                                    </p>
                                                    <div className="flex items-center gap-0.5 shrink-0">
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => { setViewInvoice(inv); setOpen(true) }}
                                                            className="h-7 w-7 p-0"
                                                            data-testid={`view-invoice-${inv.id}`}
                                                        >
                                                            <Eye className="h-3 w-3" />
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => generatePDF(inv)}
                                                            className="h-7 w-7 p-0"
                                                            data-testid={`pdf-invoice-${inv.id}`}
                                                        >
                                                            <Download className="h-3 w-3" />
                                                        </Button>
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => shareInvoice(inv)}
                                                            disabled={sharingInvoiceId === inv.id}
                                                            className="h-7 w-7 p-0"
                                                            data-testid={`share-invoice-${inv.id}`}
                                                        >
                                                            {sharingInvoiceId === inv.id ? (
                                                                <Loader2 className="h-3 w-3 animate-spin" />
                                                            ) : (
                                                                <Share2 className="h-3 w-3" />
                                                            )}
                                                        </Button>
                                                    </div>
                                                </div>

                                                <div className="flex flex-wrap gap-1 pt-1 border-t">
                                                    {inv.payment_status === "pending" && (
                                                        <>
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => addPayment(inv)}
                                                                className="h-7 px-2 text-[11px] text-green-600"
                                                                data-testid={`pay-invoice-${inv.id}`}
                                                            >
                                                                Pay
                                                            </Button>
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => handleMarkAsPaid(inv.id)}
                                                                className="h-7 px-2 text-[11px]"
                                                                data-testid={`complete-invoice-${inv.id}`}
                                                            >
                                                                Done
                                                            </Button>
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => updateInvoiceStatus(inv.id, "cancelled")}
                                                                className="h-7 px-2 text-[11px] text-destructive border-destructive/30 hover:bg-destructive/10"
                                                                data-testid={`cancel-invoice-${inv.id}`}
                                                            >
                                                                Cancel
                                                            </Button>
                                                        </>
                                                    )}

                                                    {inv.payment_status === "partial" && (
                                                        <>
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => addPayment(inv)}
                                                                className="h-7 px-2 text-[11px] text-green-600"
                                                                data-testid={`pay-invoice-${inv.id}`}
                                                            >
                                                                Pay
                                                            </Button>
                                                            <Button
                                                                size="sm"
                                                                variant="outline"
                                                                onClick={() => handleMarkAsPaid(inv.id)}
                                                                className="h-7 px-2 text-[11px]"
                                                                data-testid={`complete-invoice-${inv.id}`}
                                                            >
                                                                Done
                                                            </Button>
                                                        </>
                                                    )}
                                                </div>
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Legacy mobile cards kept disabled */}
                        {false && (
                        <div className="space-y-2">
                            <h3 className="text-sm font-semibold px-1">All Invoices ({filteredInvoices.length})</h3>
                            {filteredInvoices.length === 0 ? (
                                <div className="text-center py-12 text-muted-foreground">
                                    No invoices found
                                </div>
                            ) : (
                                <div className="space-y-2">
                                    {filteredInvoices.filter((inv) => inv && inv.id).map((inv) => (
                                        <Card key={inv.id} className="p-3 max-h-24" data-testid={`invoice-${inv.id}`}>
                                            {/* Row 1: Invoice # + Total */}
                                            <div className="flex items-center justify-between mb-1">
                                                <span className="text-xs font-semibold">{inv.invoice_number}</span>
                                                <div className="text-right">
                                                    <span className="text-sm font-bold">{formatDisplayCurrency(getInvoiceDisplayTotal(inv))}</span>
                                                    {Number(inv.advance_used || 0) > 0 && (
                                                        <p className="text-[10px] font-medium text-emerald-600">
                                                            Advance Used: {formatDisplayCurrency(inv.advance_used)}
                                                        </p>
                                                    )}
                                                </div>
                                            </div>

                                            {/* Row 2: Customer Name */}
                                            <p className="text-xs text-muted-foreground truncate mb-1">{inv.customer_name}</p>

                                            {/* Row 3: Date + Status + Actions */}
                                            <div className="flex items-center justify-between">
                                                <div className="flex items-center gap-3 text-xs text-muted-foreground">
                                                    <span>{new Date(inv.created_at).toLocaleDateString()}</span>
                                                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${inv.payment_status === "paid" ? "bg-green-100 text-green-800" :
                                                            inv.payment_status === "pending" ? "bg-yellow-100 text-yellow-800" :
                                                                inv.payment_status === "partial" ? "bg-blue-100 text-blue-800" :
                                                                    "bg-gray-100 text-gray-800"
                                                        }`}>
                                                        {inv.payment_status.toUpperCase()}
                                                    </span>
                                                </div>
                                                <div className="flex gap-1">
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        onClick={() => { setViewInvoice(inv); setOpen(true) }}
                                                        className="h-7 px-2 text-xs"
                                                        data-testid={`view-invoice-${inv.id}`}
                                                    >
                                                        <Eye className="h-3 w-3" />
                                                    </Button>
                                                    <Button
                                                        variant="ghost"
                                                        size="sm"
                                                        onClick={() => generatePDF(inv)}
                                                        className="h-7 px-2 text-xs"
                                                        data-testid={`pdf-invoice-${inv.id}`}
                                                    >
                                                        <Download className="h-3 w-3" />
                                                    </Button>
                                                    {(inv.payment_status === "partial" || inv.payment_status === "pending") && inv.balance_amount > 0 && (
                                                        <Button
                                                            variant="ghost"
                                                            size="sm"
                                                            onClick={() => addPayment(inv)}
                                                            className="h-7 px-2 text-xs text-green-600"
                                                            data-testid={`pay-invoice-${inv.id}`}
                                                        >
                                                            Pay
                                                        </Button>
                                                    )}
                                                </div>
                                            </div>
                                        </Card>
                                    ))}
                                </div>
                            )}
                        </div>
                        )}

                        {rangeFilter !== "today" && (
                        <div className="flex items-center justify-between pt-2">
                            <Button
                                variant="outline"
                                size="sm"
                                disabled={page <= 1}
                                onClick={() => setPage(page - 1)}
                                className="h-8 px-3 text-xs"
                                data-testid="previous-page-btn"
                            >
                                Previous
                            </Button>
                            <span className="text-xs text-muted-foreground">
                                Page {pagination.page} of {pagination.total_pages}
                            </span>
                            <Button
                                variant="outline"
                                size="sm"
                                disabled={page >= pagination.total_pages}
                                onClick={() => setPage(page + 1)}
                                className="h-8 px-3 text-xs"
                                data-testid="next-page-btn"
                            >
                                Next
                            </Button>
                        </div>
                        )}
                    </div>
                )}
            </div>


           {activeTab === "create" && lineItems.length > 0 && (
  <div
    className="fixed bottom-0 left-0 right-0 bg-background border-t shadow-xl z-50"
    style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
  >
    <div className="px-4 py-4 max-w-2xl mx-auto space-y-3">

      {/* Summary Row */}
      <div className="flex items-center justify-between">
        <div className="flex gap-5 text-sm">
          <span className="text-muted-foreground">
            Subtotal: <strong className="text-base">{formatDisplayCurrency(subtotal)}</strong>
          </span>
          <span className="text-muted-foreground">
            GST: <strong className="text-base">{formatDisplayCurrency(gstAmount)}</strong>
          </span>
        </div>

        <span className="text-2xl font-bold text-primary">
          {"\u20B9"}{formatCurrency(payableTotal)}
        </span>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-2">
        <Button
          variant="outline"
          onClick={resetForm}
          className="flex-1 h-11 text-sm font-medium"
        >
          Reset
        </Button>

        <Button
          variant="outline"
          onClick={() => handleSaveDraft(false)}
          className="flex-1 h-11 text-sm font-medium"
        >
          Save Draft
        </Button>

        <Button
          onClick={() => setShowPaymentDrawer(true)}
          className="flex-1 h-11 text-sm font-semibold"
        >
          Proceed ->
        </Button>
      </div>
    </div>
  </div>
)}

            {/* PAYMENT DRAWER */}
<Sheet open={showPaymentDrawer} onOpenChange={setShowPaymentDrawer}>
<SheetContent
  side="bottom"
  className="flex flex-col max-h-[90dvh] h-auto"
  style={{ paddingBottom: "env(safe-area-inset-bottom)" }}
>
    <SheetHeader>
      <SheetTitle>Payment & Finalize</SheetTitle>
    </SheetHeader>

    {/* 🔥 SCROLLABLE CONTENT */}
    <div
  className="flex-1 overflow-y-auto px-3 py-3 space-y-3 text-sm"
  style={{ WebkitOverflowScrolling: "touch" }}
>

      {/* STATUS + MODE */}
      {isAdvanceCovered ? (
        <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/10 p-3">
          <p className="text-sm font-semibold text-emerald-300">
            Advance fully covers this invoice
          </p>
          <p className="text-xs text-emerald-100/80">
            Payment status and payment mode are not required.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2">
          <div>
            <Label className="text-xs">Status*</Label>
            <Select value={normalizedPaymentStatus} onValueChange={(value) => setPaymentStatus(normalizePaymentStatus(value))}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Status" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="pending">Pending</SelectItem>
                <SelectItem value="paid">Paid</SelectItem>
                <SelectItem value="partial">Partial</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div>
            <Label className="text-xs">Mode*</Label>
            <Select value={normalizedPaymentMode} onValueChange={(value) => setPaymentMode(normalizePaymentMode(value))}>
              <SelectTrigger className="h-9">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="cash">Cash</SelectItem>
                <SelectItem value="upi">UPI</SelectItem>
                <SelectItem value="bank">Bank</SelectItem>
                <SelectItem value="cheque">Cheque</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
      )}

      {/* GST */}
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            checked={gstEnabled}
            onChange={(e) => setGstEnabled(e.target.checked)}
            className="w-4 h-4 accent-primary"
          />
          GST
        </label>

        {gstEnabled && (
          <Input
            type="number"
            value={gstRate}
            onChange={(e) => setGstRate(Number(e.target.value))}
            className="h-8 w-20 text-xs"
          />
        )}
      </div>

      {/* DISCOUNT */}
      <div>
        <Label className="text-xs">Discount</Label>
        <Input
          type="number"
          value={discount}
          onChange={(e) => {
            let val = e.target.value
            if (val.length > 1 && val.startsWith("0")) {
              val = val.replace(/^0+/, "")
            }
            setDiscount(val)
          }}
          className="h-9"
        />
      </div>

      {/* PARTIAL */}
      {!isAdvanceCovered && normalizedPaymentStatus === "partial" && (
        <div>
          <Label className="text-xs">Paid</Label>
          <Input
            type="number"
            value={paidAmount}
            onChange={(e) => setPaidAmount(Number(e.target.value))}
            className="h-9"
          />
        </div>
      )}

      {/* EXTRA CHARGES (FULL SCROLL SUPPORT) */}
      <div className="space-y-2">
        <div className="flex justify-between items-center">
          <Label className="text-xs">Extra</Label>
          <Button
            size="sm"
            variant="ghost"
            onClick={addAdditionalCharge}
            className="h-6 px-2 text-xs"
          >
            +
          </Button>
        </div>

        {additionalCharges.map((c, i) => (
          <div key={i} className="flex gap-2">
            <Input
              placeholder="Label"
              value={c.label}
              onChange={(e) =>
                updateAdditionalCharge(i, "label", e.target.value)
              }
              className="h-9 text-xs"
            />
            <Input
              type="number"
              placeholder={RUPEE_SYMBOL}
              value={c.amount}
              onChange={(e) =>
                updateAdditionalCharge(i, "amount", e.target.value)
              }
              className="h-9 text-xs w-24"
            />
            <button onClick={() => removeAdditionalCharge(i)}>
              <X className="w-4 h-4" />
            </button>
          </div>
        ))}
      </div>

      {/* 🔥 EXTRA SPACE SO LAST ITEM NOT HIDDEN */}
      <div className="h-16" />

    </div>

    {/* 🔥 FIXED SUMMARY */}
    <div className="border-t px-3 py-2 text-sm space-y-1 bg-background">

      <div className="flex justify-between">
        <span>Subtotal</span>
        <span>{formatDisplayCurrency(subtotal)}</span>
      </div>

      {additionalTotal > 0 && (
        <div className="flex justify-between">
          <span>Extra</span>
          <span>{formatDisplayCurrency(additionalTotal)}</span>
        </div>
      )}

      {gstEnabled && (
        <div className="flex justify-between">
          <span>GST</span>
          <span>{formatDisplayCurrency(gstAmount)}</span>
        </div>
      )}

      {discount > 0 && (
        <div className="flex justify-between text-red-500">
          <span>Discount</span>
          <span>-{formatDisplayCurrency(discount)}</span>
        </div>
      )}

      <div className="flex justify-between font-bold text-base border-t pt-1">
        <span>Total</span>
        <span>{formatDisplayCurrency(total)}</span>
      </div>

      {appliedAdvance > 0 && (
        <div className="flex justify-between text-emerald-600 text-xs">
          <span>Advance Used</span>
          <span>-{formatDisplayCurrency(appliedAdvance)}</span>
        </div>
      )}

      <div className="flex justify-between font-semibold text-sm">
        <span>Payable</span>
        <span>{formatDisplayCurrency(payableTotal)}</span>
      </div>

      {!isAdvanceCovered && normalizedPaymentStatus === "partial" && (
        <div className="flex justify-between text-yellow-600 text-xs">
          <span>Balance Amount</span>
          <span>{formatDisplayCurrency(remainingBalance)}</span>
        </div>
      )}
    </div>

    {/* BUTTONS */}
    <SheetFooter className="p-3 border-t bg-background">
      <div className="flex gap-2 w-full">
        <Button variant="outline" className="flex-1" onClick={() => handleSaveDraft(false)}>
          Draft
        </Button>
        <Button className="flex-1" onClick={handleCreateInvoice}>
          {creatingInvoice ? "..." : "Finish"}
        </Button>
      </div>
    </SheetFooter>

  </SheetContent>
</Sheet>

            {/* SUCCESS MODAL */}
            <Dialog open={showSuccessModal} onOpenChange={setShowSuccessModal}>
                <DialogContent className="sm:max-w-md">
                    <DialogHeader>
                        <div className="flex justify-center mb-4">
                            <div className="bg-green-100 p-3 rounded-full">
                                <CheckCircle2 className="h-12 w-12 text-green-600" />
                            </div>
                        </div>
                        <DialogTitle className="text-center text-xl">Invoice Created Successfully!</DialogTitle>
                    </DialogHeader>

                    <div className="text-center space-y-2 py-4">
                        <p className="text-sm text-muted-foreground">Invoice Number</p>
                        <p className="text-2xl font-bold">{createdInvoice?.invoice_number}</p>
                        <p className="text-sm text-muted-foreground">Total Amount</p>
                        <p className="text-3xl font-bold text-primary">{formatDisplayCurrency(createdInvoice?.total || 0)}</p>
                    </div>

                    <DialogFooter className="flex flex-col sm:flex-row gap-2">
                        <Button
                            variant="outline"
                            onClick={() => {
                                setShowSuccessModal(false)
                                resetForm()
                                setActiveTab("create")
                            }}
                            className="flex-1"
                            data-testid="create-new-invoice-btn"
                        >
                            <Plus className="h-4 w-4 mr-2" />
                            Create New Invoice
                        </Button>
                        <Button
                            onClick={() => {
                                setShowSuccessModal(false)
                                setActiveTab("list")
                            }}
                            className="flex-1"
                            data-testid="view-invoice-btn"
                        >
                            <Eye className="h-4 w-4 mr-2" />
                            View Invoice
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* VIEW INVOICE DIALOG */}
            {viewInvoice && (
                <Dialog open={open} onOpenChange={setOpen}>
                    <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
                        <DialogHeader>
                            <DialogTitle>Invoice #{viewInvoice.invoice_number}</DialogTitle>
                        </DialogHeader>

                        <div className="space-y-4">
                            {/* Customer Info */}
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <p className="text-muted-foreground">Customer Name</p>
                                    <p className="font-medium">{viewInvoice.customer_name}</p>
                                </div>
                                <div>
                                    <p className="text-muted-foreground">Phone</p>
                                    <p className="font-medium">{viewInvoice.customer_phone || "N/A"}</p>
                                </div>
                                <div>
                                    <p className="text-muted-foreground">Date</p>
                                    <p className="font-medium">{new Date(viewInvoice.created_at).toLocaleDateString()}</p>
                                </div>
                                <div>
                                    <p className="text-muted-foreground">Status</p>
                                    <span className={`inline-block px-2 py-1 rounded text-xs font-medium ${viewInvoice.payment_status === "paid" ? "bg-green-100 text-green-800" :
                                            viewInvoice.payment_status === "pending" ? "bg-yellow-100 text-yellow-800" :
                                                viewInvoice.payment_status === "partial" ? "bg-blue-100 text-blue-800" :
                                                    "bg-gray-100 text-gray-800"
                                        }`}>
                                        {viewInvoice.payment_status.toUpperCase()}
                                    </span>
                                </div>
                            </div>

                            {/* Items */}
                            <div>
                                <h4 className="font-semibold mb-2">Items</h4>
                                <div className="border rounded-lg overflow-hidden">
                                    <table className="w-full text-sm">
                                        <thead className="bg-muted">
                                            <tr>
                                                <th className="text-left p-2">Product</th>
                                                <th className="text-center p-2">Qty</th>
                                                <th className="text-right p-2">Price</th>
                                                <th className="text-right p-2">Total</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {viewInvoice.items.map((item, idx) => (
                                                <tr key={idx} className="border-t">
                                                    <td className="p-2">
                                                        <div>
                                                            <p className="font-medium">{item.product_name}</p>
                                                            {(item.color || item.size || item.v_sku) && (
                                                                <p className="text-xs text-muted-foreground">
                                                                    {item.color && `Color: ${item.color}`}
                                                                    {item.size && ` • Size: ${item.size}`}
                                                                    {item.v_sku && ` • SKU: ${item.v_sku}`}
                                                                </p>
                                                            )}
                                                        </div>
                                                    </td>
                                                    <td className="text-center p-2">{item.quantity}</td>
                                                    <td className="text-right p-2">{formatDisplayCurrency(item.price)}</td>
                                                    <td className="text-right p-2 font-medium">{formatDisplayCurrency(item.total)}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                            {/* Totals */}
                            <div className="border-t pt-4 space-y-2 text-sm">
                                <div className="flex justify-between">
                                    <span>Subtotal:</span>
                                    <span className="font-medium">{formatDisplayCurrency(viewInvoice.subtotal)}</span>
                                </div>

                                {Array.isArray(viewInvoice.additional_charges) &&
                                    viewInvoice.additional_charges
                                        .filter((c) => Number(c.amount) > 0)
                                        .map((c, i) => (
                                            <div key={i} className="flex justify-between">
                                                <span>{c.label || "Additional Charge"}:</span>
                                                <span className="font-medium">{formatDisplayCurrency(c.amount)}</span>
                                            </div>
                                        ))}

                                <div className="flex justify-between">
                                    <span>GST:</span>
                                    <span className="font-medium">{formatDisplayCurrency(viewInvoice.gst_amount)}</span>
                                </div>
                                <div className="flex justify-between">
                                    <span>Discount:</span>
                                    <span className="font-medium">-{formatDisplayCurrency(viewInvoice.discount)}</span>
                                </div>
                                <div className="flex justify-between text-lg font-bold border-t pt-2">
                                    <span>Total:</span>
                                    <span>{formatDisplayCurrency(getInvoiceDisplayTotal(viewInvoice))}</span>
                                </div>
                                {Number(viewInvoice.advance_used || 0) > 0 && (
                                    <div className="flex justify-between text-emerald-600">
                                        <span>Advance Used:</span>
                                        <span className="font-medium">{formatDisplayCurrency(viewInvoice.advance_used || 0)}</span>
                                    </div>
                                )}

                                {viewInvoice.payment_status === "paid" && (
                                    <div className="bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-green-900">
                                        <strong>Status: ✓ FULLY PAID</strong>
                                    </div>
                                )}

                                {viewInvoice.payment_status === "partial" && (
                                    <>
                                        <div className="flex justify-between text-green-600">
                                            <span>Paid Amount:</span>
                                            <span className="font-medium">{formatDisplayCurrency(viewInvoice.paid_amount || 0)}</span>
                                        </div>
                                        <div className="flex justify-between text-red-600">
                                            <span>Balance Remaining:</span>
                                            <span className="font-medium">{formatDisplayCurrency(viewInvoice.balance_amount || 0)}</span>
                                        </div>
                                    </>
                                )}
                            </div>
                        </div>

                        <DialogFooter>
                            <Button
                                variant="outline"
                                onClick={() => generatePDF(viewInvoice)}
                                data-testid="download-pdf-btn"
                            >
                                <Download className="h-4 w-4 mr-2" />
                                Download PDF
                            </Button>
                        </DialogFooter>
                    </DialogContent>
                </Dialog>
            )}

            <Dialog open={comboModal} onOpenChange={setComboModal}>
                <DialogContent className="max-w-md">
                    <DialogHeader>
                        <DialogTitle>Select Combo</DialogTitle>
                    </DialogHeader>

                    <div className="max-h-[320px] overflow-y-auto space-y-2">
                        {combos.length === 0 ? (
                            <div className="text-sm text-muted-foreground py-6 text-center">
                                No combos available
                            </div>
                        ) : (
                            combos.map((combo) => (
                                <button
                                    key={combo.id}
                                    onClick={() => addComboToInvoice(combo)}
                                    className="w-full p-3 border rounded-xl text-left hover:bg-muted transition-colors"
                                    data-testid={`combo-option-${combo.id}`}
                                >
                                    <p className="font-medium">{combo.name}</p>
                                    <p className="text-xs text-muted-foreground">
                                        {formatDisplayCurrency(combo.price)} • {combo.items?.length || 0} items
                                    </p>
                                </button>
                            ))
                        )}
                    </div>
                </DialogContent>
            </Dialog>

            {/* VARIANT SELECTION DIALOG */}
            <Dialog
                open={showVariantDialog}
                onOpenChange={(open) => {
                    setShowVariantDialog(open)
                    if (!open && pendingCombo) {
                        setPendingCombo(null)
                        setSelectedProductForVariant(null)
                    }
                }}
            >
                <DialogContent className="max-w-md max-h-[80vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle>Select Variant</DialogTitle>
                        <p className="text-sm text-muted-foreground">{selectedProductForVariant?.name} - Choose your option</p>
                    </DialogHeader>

                    <div className="space-y-2" onKeyDown={handleVariantKeyDown} tabIndex={0}>
                        {selectedProductForVariant?.variants?.map((variant, index) => (
                            <button
                                key={index}
                                id={`variant-${index}`}
                                onClick={(e) => {
                                    e.preventDefault()
                                    addSelectedProductToInvoice(selectedProductForVariant, variant)
                                }}
                                className={`w-full flex items-center gap-3 p-3 rounded-lg border transition-all text-left ${selectedVariantIndex === index
                                        ? "border-primary bg-primary/10 shadow-lg"
                                        : "bg-card hover:bg-primary/5 hover:border-primary hover:shadow"
                                    }`}
                                data-testid={`variant-option-${index}`}
                            >
                                <img
                                    src={variant.image_url || variant.v_image_url || selectedProductForVariant.images?.[0] || "/placeholder.png"}
                                    alt={variant.variant_name}
                                    className="w-12 h-12 object-cover rounded"
                                />
                                <div className="flex-1">
                                    <p className="text-sm font-medium">{selectedProductForVariant.name}</p>
                                    <div className="text-xs text-muted-foreground space-y-0.5">
                                        <p>{variant.v_sku}</p>
                                        {variant.variant_name && <p>Variant: {variant.variant_name}</p>}
                                        {variant.color && <p>Color: {variant.color}</p>}
                                        {variant.size && <p>Size: {variant.size}</p>}
                                    </div>
                                </div>
                                <div className="text-right">
                                    <p className="text-sm font-bold">₹{variant.v_selling_price || selectedProductForVariant.selling_price}</p>
                                    {variant.stock && (
                                        <p className="text-xs text-muted-foreground">Stock: {variant.stock}</p>
                                    )}
                                </div>
                            </button>
                        ))}
                    </div>

                    <DialogFooter>
                        <Button variant="outline" onClick={() => setShowVariantDialog(false)}>
                            Cancel
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* MANUAL ITEM DIALOG */}
            <Dialog open={showManualDialog} onOpenChange={setShowManualDialog}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Add Manual Item</DialogTitle>
                    </DialogHeader>

                    <div className="space-y-4">
                        <div>
                            <Label>Item Name*</Label>
                            <Input
                                value={manualItem.name}
                                onChange={(e) => setManualItem({ ...manualItem, name: e.target.value })}
                                placeholder="e.g., Service Charge"
                                data-testid="manual-item-name-input"
                            />
                        </div>
                        <div>
                            <Label>Price (₹)*</Label>
                            <Input
                                type="number"
                                value={manualItem.price}
                                onChange={(e) => setManualItem({ ...manualItem, price: e.target.value })}
                                placeholder="0.00"
                                data-testid="manual-item-price-input"
                            />
                        </div>
                        <div>
                            <Label>Quantity</Label>
                            <Input
                                type="number"
                                value={manualItem.quantity}
                                onChange={(e) => setManualItem({ ...manualItem, quantity: e.target.value })}
                                min="1"
                                data-testid="manual-item-qty-input"
                            />
                        </div>
                    </div>

                    <DialogFooter>
                        <Button variant="outline" onClick={() => setShowManualDialog(false)}>
                            Cancel
                        </Button>
                        <Button onClick={addManualItemToInvoice} data-testid="add-manual-item-confirm-btn">
                            Add Item
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* MARK AS PAID DIALOG */}
            <Dialog open={isPaidModeDialogOpen} onOpenChange={setIsPaidModeDialogOpen}>
                <DialogContent>
                    <DialogHeader>
                        <DialogTitle>Mark Invoice as Paid</DialogTitle>
                        <p className="text-sm text-muted-foreground">How was the payment received?</p>
                    </DialogHeader>

                    <div className="space-y-4">
                        <div>
                            <Label>Payment Mode*</Label>
                            <Select value={paidPaymentMode} onValueChange={setPaidPaymentMode}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="cash">💵 Cash</SelectItem>
                                    <SelectItem value="upi">📱 UPI</SelectItem>
                                    <SelectItem value="bank">🏦 Bank Transfer</SelectItem>
                                    <SelectItem value="cheque">📋 Cheque</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {paidPaymentMode === "upi" && (
                            <div>
                                <Label>UPI ID</Label>
                                <Input
                                    value={paidPaymentRef}
                                    onChange={(e) => setPaidPaymentRef(e.target.value)}
                                    placeholder="user@upi"
                                />
                            </div>
                        )}

                        {paidPaymentMode === "bank" && (
                            <div>
                                <Label>Transaction ID</Label>
                                <Input
                                    value={paidPaymentRef}
                                    onChange={(e) => setPaidPaymentRef(e.target.value)}
                                    placeholder="TXN123456"
                                />
                            </div>
                        )}

                        {paidPaymentMode === "cheque" && (
                            <div>
                                <Label>Cheque Number</Label>
                                <Input
                                    value={paidPaymentRef}
                                    onChange={(e) => setPaidPaymentRef(e.target.value)}
                                    placeholder="123456"
                                />
                            </div>
                        )}
                    </div>

                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsPaidModeDialogOpen(false)}>
                            Cancel
                        </Button>
                        <Button onClick={confirmMarkAsPaid}>
                            Confirm Paid
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            {/* PAYMENT MODAL */}
            <Dialog open={isPaymentModalOpen} onOpenChange={setIsPaymentModalOpen}>
                <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
                    <DialogHeader>
                        <DialogTitle>Add Payment - {viewInvoice?.invoice_number}</DialogTitle>
                    </DialogHeader>

                    <div className="space-y-4">
                        <div className="bg-muted p-3 rounded-lg space-y-1 text-sm">
                            <div className="flex justify-between">
                                <span>Total:</span>
                                <span className="font-semibold">{formatDisplayCurrency(viewInvoice?.total || 0)}</span>
                            </div>
                            <div className="flex justify-between text-green-600">
                                <span>Paid:</span>
                                <span className="font-semibold">{formatDisplayCurrency(viewInvoice?.paid_amount || 0)}</span>
                            </div>
                            <div className="flex justify-between text-red-600">
                                <span>Balance:</span>
                                <span className="font-semibold">{formatDisplayCurrency(viewInvoice?.balance_amount || 0)}</span>
                            </div>
                        </div>

                        <div>
                            <Label>Payment Amount ({RUPEE_SYMBOL})*</Label>
                            <Input
                                type="number"
                                value={paymentAmount}
                                onChange={(e) => setPaymentAmount(e.target.value)}
                                placeholder="0.00"
                                className="text-lg font-bold"
                            />
                        </div>

                        <div>
                            <Label>Payment Mode*</Label>
                            <Select value={paymentModeSelected} onValueChange={setPaymentModeSelected}>
                                <SelectTrigger>
                                    <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                    <SelectItem value="cash">💵 Cash</SelectItem>
                                    <SelectItem value="upi">📱 UPI</SelectItem>
                                    <SelectItem value="bank">🏦 Bank Transfer</SelectItem>
                                    <SelectItem value="cheque">📋 Cheque</SelectItem>
                                </SelectContent>
                            </Select>
                        </div>

                        {paymentModeSelected === "upi" && (
                            <div>
                                <Label>UPI ID</Label>
                                <Input
                                    value={paymentReference}
                                    onChange={(e) => setPaymentReference(e.target.value)}
                                    placeholder="user@upi"
                                />
                            </div>
                        )}

                        {paymentModeSelected === "bank" && (
                            <div>
                                <Label>Transaction ID</Label>
                                <Input
                                    value={paymentReference}
                                    onChange={(e) => setPaymentReference(e.target.value)}
                                    placeholder="TXN123456"
                                />
                            </div>
                        )}

                        {paymentModeSelected === "cheque" && (
                            <div>
                                <Label>Cheque Number</Label>
                                <Input
                                    value={paymentReference}
                                    onChange={(e) => setPaymentReference(e.target.value)}
                                    placeholder="123456"
                                />
                            </div>
                        )}

                        {paymentHistory.length > 0 && (
                            <div>
                                <h4 className="font-semibold mb-2 text-sm">Payment History</h4>
                                <div className="space-y-2">
                                    {paymentHistory.map((payment) => (
                                        <div key={payment.id} className="flex items-center justify-between p-2 bg-muted rounded text-sm">
                                            <div>
                                                <p className="font-medium">
                                                    {formatDisplayCurrency(payment.amount)} via {payment.payment_mode.toUpperCase()}
                                                </p>
                                                <p className="text-xs text-muted-foreground">
                                                    {new Date(payment.created_at).toLocaleDateString('en-IN', {
                                                        day: 'numeric',
                                                        month: 'short',
                                                        hour: '2-digit',
                                                        minute: '2-digit'
                                                    })}
                                                </p>
                                                {payment.reference && (
                                                    <p className="text-xs text-muted-foreground">{payment.reference}</p>
                                                )}
                                            </div>
                                            <p className="text-[10px] font-medium text-muted-foreground">
                                                Audit locked
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>

                    <DialogFooter>
                        <Button variant="outline" onClick={() => setIsPaymentModalOpen(false)}>
                            Cancel
                        </Button>
                        <Button onClick={handleAddPayment}>
                            Add Payment
                        </Button>
                    </DialogFooter>
                </DialogContent>
            </Dialog>

            <Dialog open={exportDialogOpen} onOpenChange={setExportDialogOpen}>
                <DialogContent className="max-w-sm rounded-3xl">
                    <DialogHeader>
                        <DialogTitle>Export Invoices</DialogTitle>
                    </DialogHeader>
                    <div className="space-y-4">
                        <Select value={exportRange} onValueChange={setExportRange}>
                            <SelectTrigger className="h-12 rounded-2xl">
                                <SelectValue placeholder="Select export range" />
                            </SelectTrigger>
                            <SelectContent>
                                <SelectItem value="today">Today</SelectItem>
                                <SelectItem value="week">This Week</SelectItem>
                                <SelectItem value="last10">Last 10 Days</SelectItem>
                                <SelectItem value="lastMonth">Last Month</SelectItem>
                            </SelectContent>
                        </Select>

                        <div className="rounded-2xl bg-muted/20 p-4 text-sm text-muted-foreground">
                            Export creates a table-style PDF using the selected date range and current status filter.
                        </div>

                        <div className="flex gap-2">
                            <Button variant="outline" className="flex-1 rounded-2xl" onClick={() => setExportDialogOpen(false)}>
                                Cancel
                            </Button>
                            <Button className="flex-1 rounded-2xl" onClick={exportInvoices} disabled={exportingInvoices}>
                                {exportingInvoices ? "Exporting..." : "Download PDF"}
                            </Button>
                        </div>
                    </div>
                </DialogContent>
            </Dialog>
            {previewImage && (
  <div
    className="fixed inset-0 z-[999] bg-black/80 flex items-center justify-center"
    onClick={() => setPreviewImage(null)}
  >
    {/* IMAGE */}
    <img
      src={previewImage}
      alt="Preview"
      className="max-h-[90vh] max-w-[90vw] rounded-lg shadow-xl"
      onClick={(e) => e.stopPropagation()} // prevent close on image click
    />

    {/* CLOSE BUTTON */}
    <button
      onClick={() => setPreviewImage(null)}
      className="absolute top-5 right-5 bg-white/10 hover:bg-white/20 text-white p-2 rounded-full"
    >
      <X className="w-6 h-6" />
    </button>
  </div>
)}
        </div>
    )
}

