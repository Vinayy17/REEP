"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { useLocation, useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { toast } from "sonner"
import { Plus, User, FileText, Search, Building2, Trash2, Edit, Phone, Wallet } from "lucide-react"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const emptyCustomer = {
  name: "",
  email: "",
  phone: "",
  address: "",
}

const emptyAdvanceForm = {
  amount: "",
  payment_mode: "cash",
  reference: "",
}

const CUSTOMERS_PER_PAGE = 30

const hasNumericBalance = (value) =>
  value !== null && value !== undefined && value !== "" && !Number.isNaN(Number(value))

export default function Customers() {
  const navigate = useNavigate()
  const location = useLocation()

  const [activeTab, setActiveTab] = useState("customers")
  const [customers, setCustomers] = useState([])
  const [suppliers, setSuppliers] = useState([])
  const [supplierInvoices, setSupplierInvoices] = useState([])
  const [searchTerm, setSearchTerm] = useState("")
  const [supplierSearch, setSupplierSearch] = useState("")
  const [viewSupplier, setViewSupplier] = useState(null)
  const [supplierLedger, setSupplierLedger] = useState(null)
  const [supplierLedgerLoading, setSupplierLedgerLoading] = useState(false)
  const [supplierLedgerSearch, setSupplierLedgerSearch] = useState("")
  const [supplierDetailTab, setSupplierDetailTab] = useState("ledger")
  const [supplierBillDialogOpen, setSupplierBillDialogOpen] = useState(false)
  const [supplierPaymentDialogOpen, setSupplierPaymentDialogOpen] = useState(false)
  const [supplierBillSupplierId, setSupplierBillSupplierId] = useState("")
  const [supplierBillAmount, setSupplierBillAmount] = useState("")
  const [supplierBillDate, setSupplierBillDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [supplierPaymentInvoiceId, setSupplierPaymentInvoiceId] = useState("")
  const [supplierPaymentAmount, setSupplierPaymentAmount] = useState("")
  const [supplierPaymentMode, setSupplierPaymentMode] = useState("cash")
  const [supplierPaymentNotes, setSupplierPaymentNotes] = useState("")
  const [supplierPaymentDate, setSupplierPaymentDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [supplierBillSearch, setSupplierBillSearch] = useState("")
  const [supplierPaymentSearch, setSupplierPaymentSearch] = useState("")
  const [openCustomer, setOpenCustomer] = useState(false)
  const [openSupplier, setOpenSupplier] = useState(false)
  const [editingCustomer, setEditingCustomer] = useState(null)
  const [viewCustomer, setViewCustomer] = useState(null)
  const [customerLedger, setCustomerLedger] = useState(null)
  const [customerLedgerLoading, setCustomerLedgerLoading] = useState(false)
  const [customerPage, setCustomerPage] = useState(1)
  const [customerSort, setCustomerSort] = useState("a-z")
  const [customerDetailTab, setCustomerDetailTab] = useState("ledger")
  const [customerLedgerSearch, setCustomerLedgerSearch] = useState("")
  const [customerInvoiceSearch, setCustomerInvoiceSearch] = useState("")
  const [customerInvoiceStatusFilter, setCustomerInvoiceStatusFilter] = useState("all")
  const [customerItemSearch, setCustomerItemSearch] = useState("")
  const [selectedCustomerInvoice, setSelectedCustomerInvoice] = useState(null)
  const [customerForm, setCustomerForm] = useState(emptyCustomer)
  const [advanceDialogOpen, setAdvanceDialogOpen] = useState(false)
  const [advanceCustomer, setAdvanceCustomer] = useState(null)
  const [advanceForm, setAdvanceForm] = useState(emptyAdvanceForm)
  const [advanceSubmitting, setAdvanceSubmitting] = useState(false)
  const [supplierForm, setSupplierForm] = useState({
    name: "",
    email: "",
    phone: "",
    address: "",
    opening_balance: 0,
  })
  useEffect(() => {
    fetchCustomers()
    fetchSuppliers()
    fetchSupplierInvoices()
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get("action") === "add") {
      setOpenCustomer(true)
      navigate("/customers", { replace: true })
    }
  }, [location.search, navigate])

  useEffect(() => {
    setCustomerPage(1)
  }, [searchTerm, customerSort, activeTab])

  const tokenHeaders = () => ({
    headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
  })

  const fetchCustomers = async () => {
    try {
      const res = await axios.get(`${API}/customers`, tokenHeaders())
      const data = Array.isArray(res.data) ? res.data : res.data?.data || []
      setCustomers(data)
      return data
    } catch {
      toast.error("Failed to load customers")
      return []
    }
  }

  const fetchSuppliers = async () => {
    try {
      const res = await axios.get(`${API}/suppliers`, tokenHeaders())
      const data = Array.isArray(res.data) ? res.data : res.data?.data || []
      setSuppliers(data)
      return data
    } catch {
      toast.error("Failed to load suppliers")
      return []
    }
  }

  const fetchSupplierInvoices = async () => {
    try {
      const res = await axios.get(`${API}/supplier-invoices`, tokenHeaders())
      const data = Array.isArray(res.data) ? res.data : res.data?.data || []
      setSupplierInvoices(data)
      return data
    } catch {
      toast.error("Failed to load supplier bills")
      return []
    }
  }

  const resetCustomerForm = () => {
    setEditingCustomer(null)
    setCustomerForm(emptyCustomer)
  }

  const submitCustomer = async (e) => {
    e.preventDefault()
    try {
      if (editingCustomer) {
        await axios.put(`${API}/customers/${editingCustomer.id}`, customerForm, tokenHeaders())
        toast.success("Customer updated successfully")
      } else {
        await axios.post(`${API}/customers`, customerForm, tokenHeaders())
        toast.success("Customer added successfully")
      }

      fetchCustomers()
      setOpenCustomer(false)
      resetCustomerForm()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to save customer")
    }
  }

  const deleteCustomer = async (customer) => {
    if (!window.confirm(`Delete ${customer.name}?`)) return

    try {
      await axios.delete(`${API}/customers/${customer.id}`, tokenHeaders())
      toast.success("Customer deleted successfully")
      fetchCustomers()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to delete customer")
    }
  }

  const addSupplier = async (e) => {
    e.preventDefault()
    try {
      await axios.post(`${API}/suppliers`, supplierForm, tokenHeaders())
      toast.success("Supplier added successfully")
      fetchSuppliers()
      setOpenSupplier(false)
      setSupplierForm({
        name: "",
        email: "",
        phone: "",
        address: "",
        opening_balance: 0,
      })
    } catch {
      toast.error("Failed to add supplier")
    }
  }

  const resetSupplierBillDialog = () => {
    setSupplierBillDialogOpen(false)
    setSupplierBillSupplierId("")
    setSupplierBillAmount("")
    setSupplierBillDate(new Date().toISOString().slice(0, 10))
    setSupplierBillSearch("")
  }

  const openSupplierBillDialogFor = (supplier = null) => {
    setSupplierBillSupplierId(supplier?.id || "")
    setSupplierBillAmount("")
    setSupplierBillDate(new Date().toISOString().slice(0, 10))
    setSupplierBillSearch(supplier ? getSupplierLookupLabel(supplier) : "")
    setSupplierBillDialogOpen(true)
  }

  const resetSupplierPaymentDialog = () => {
    setSupplierPaymentDialogOpen(false)
    setSupplierPaymentInvoiceId("")
    setSupplierPaymentAmount("")
    setSupplierPaymentMode("cash")
    setSupplierPaymentNotes("")
    setSupplierPaymentDate(new Date().toISOString().slice(0, 10))
    setSupplierPaymentSearch("")
  }

  const openSupplierPaymentDialogFor = (supplier = null) => {
    const matchingInvoices = pendingSupplierInvoices.filter(
      (invoice) => !supplier || invoice.supplier_id === supplier.id
    )
    const firstInvoice = matchingInvoices[0] || null

    setSupplierPaymentInvoiceId(firstInvoice?.id || "")
    setSupplierPaymentAmount(firstInvoice ? String(Number(firstInvoice.pending_amount || 0)) : "")
    setSupplierPaymentMode("cash")
    setSupplierPaymentNotes("")
    setSupplierPaymentDate(new Date().toISOString().slice(0, 10))
    setSupplierPaymentSearch(firstInvoice ? getSupplierInvoiceLookupLabel(firstInvoice) : "")
    setSupplierPaymentDialogOpen(true)
  }

  const refreshSupplierSection = async (supplierId = null) => {
    const [refreshedSuppliers] = await Promise.all([fetchSuppliers(), fetchSupplierInvoices()])

    if (supplierId && viewSupplier?.id === supplierId) {
      const refreshedSupplier =
        refreshedSuppliers.find((supplier) => supplier.id === supplierId) ||
        viewSupplier
      await openSupplierDetails(refreshedSupplier)
    }
  }

  const addSupplierBill = async () => {
    if (!supplierBillSupplierId || !supplierBillAmount) {
      toast.error("Select supplier and enter bill amount")
      return
    }

    try {
      await axios.post(
        `${API}/suppliers/${supplierBillSupplierId}/invoices`,
        {
          total_amount: Number(supplierBillAmount),
          invoice_date: supplierBillDate || null,
        },
        tokenHeaders()
      )

      toast.success("Supplier bill added")
      resetSupplierBillDialog()
      await refreshSupplierSection(supplierBillSupplierId)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add supplier bill")
    }
  }

  const paySupplierBill = async () => {
    if (!supplierPaymentInvoiceId || !supplierPaymentAmount) {
      toast.error("Select a pending bill and enter amount")
      return
    }

    const targetInvoice = supplierInvoices.find((invoice) => invoice.id === supplierPaymentInvoiceId)

    try {
      await axios.post(
        `${API}/supplier-invoices/${supplierPaymentInvoiceId}/pay`,
        {
          amount: Number(supplierPaymentAmount),
          payment_mode: supplierPaymentMode,
          payment_date: supplierPaymentDate || null,
          notes: supplierPaymentNotes || null,
        },
        tokenHeaders()
      )

      toast.success("Supplier payment recorded")
      resetSupplierPaymentDialog()
      await refreshSupplierSection(targetInvoice?.supplier_id || null)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to pay supplier bill")
    }
  }

  const handleCreateInvoice = (customer) => {
    navigate("/invoices", {
      state: {
        customer: {
          id: customer.id,
          name: customer.name,
          phone: customer.phone,
          email: customer.email,
          address: customer.address,
          current_balance: hasNumericBalance(customer.current_balance) ? Number(customer.current_balance) : 0,
        },
      },
    })
  }

  const filteredCustomers = customers.filter((c) =>
    `${c.name} ${c.email} ${c.phone}`.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const sortedCustomers = [...filteredCustomers].sort((a, b) => {
    const left = String(a?.name || "").toLowerCase()
    const right = String(b?.name || "").toLowerCase()

    if (customerSort === "z-a") {
      return right.localeCompare(left)
    }

    return left.localeCompare(right)
  })

  const customerTotalPages = Math.max(Math.ceil(sortedCustomers.length / CUSTOMERS_PER_PAGE), 1)
  const safeCustomerPage = Math.min(customerPage, customerTotalPages)
  const paginatedCustomers = sortedCustomers.slice(
    (safeCustomerPage - 1) * CUSTOMERS_PER_PAGE,
    safeCustomerPage * CUSTOMERS_PER_PAGE,
  )

  const filteredSuppliers = suppliers.filter((s) =>
    `${s.name} ${s.phone} ${s.email} ${s.address}`.toLowerCase().includes(supplierSearch.toLowerCase())
  )
  const sortedFilteredSuppliers = [...filteredSuppliers].sort((leftSupplier, rightSupplier) =>
    String(leftSupplier?.name || "").localeCompare(String(rightSupplier?.name || ""), "en", {
      sensitivity: "base",
    })
  )
  const supplierBillOptions = [...suppliers]
    .filter((supplier) =>
      matchesSearchText(
        [
          supplier?.name,
          supplier?.phone,
          supplier?.email,
          supplier?.address,
          getSupplierLookupLabel(supplier),
        ],
        supplierBillSearch
      )
    )
    .sort((leftSupplier, rightSupplier) =>
      String(leftSupplier?.name || "").localeCompare(String(rightSupplier?.name || ""), "en", {
        sensitivity: "base",
      })
    )
  const pendingSupplierInvoices = supplierInvoices.filter(
    (invoice) => Number(invoice.pending_amount || 0) > 0
  )
  const pendingSupplierInvoiceOptions = [...pendingSupplierInvoices]
    .filter((invoice) =>
      matchesSearchText(
        [
          invoice?.bill_number,
          invoice?.supplier_name,
          invoice?.invoice_date,
          invoice?.pending_amount,
          invoice?.paid_amount,
          invoice?.status,
          getSupplierInvoiceLookupLabel(invoice),
        ],
        supplierPaymentSearch
      )
    )
    .sort((leftInvoice, rightInvoice) => {
      const supplierCompare = String(leftInvoice?.supplier_name || "").localeCompare(
        String(rightInvoice?.supplier_name || ""),
        "en",
        { sensitivity: "base" }
      )

      if (supplierCompare !== 0) return supplierCompare

      return String(rightInvoice?.invoice_date || "").localeCompare(String(leftInvoice?.invoice_date || ""))
    })
  const supplierOverview = {
    totalBill: sortedFilteredSuppliers.reduce((sum, supplier) => sum + Number(supplier.total_bill || 0), 0),
    paid: sortedFilteredSuppliers.reduce((sum, supplier) => sum + Number(supplier.paid_amount || 0), 0),
    pending: sortedFilteredSuppliers.reduce((sum, supplier) => sum + Number(supplier.pending_amount || 0), 0),
    count: sortedFilteredSuppliers.length,
  }
  const selectedSupplierForBill =
    suppliers.find((supplier) => supplier.id === supplierBillSupplierId) || null
  const selectedSupplierPaymentInvoice =
    supplierInvoices.find((invoice) => invoice.id === supplierPaymentInvoiceId) || null

  const renderBalanceLabel = (balance) => {
    const amount = Number(balance || 0)
    if (amount < 0) return `Advance Available: ₹${Math.abs(amount).toLocaleString("en-IN")}`
    if (amount > 0) return `Pending Amount: ₹${amount.toLocaleString("en-IN")}`
    return "No pending or advance"
  }

  function formatMoney(amount) {
    return Number(amount || 0).toLocaleString("en-IN", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    })
  }
  function getSupplierLookupLabel(supplier) {
    return [supplier?.name, supplier?.phone, supplier?.email].filter(Boolean).join(" | ")
  }

  function getSupplierInvoiceLookupLabel(invoice) {
    return [
      invoice?.bill_number || invoice?.id,
      invoice?.supplier_name,
      `Pending Rs. ${formatMoney(invoice?.pending_amount)}`,
    ]
      .filter(Boolean)
      .join(" | ")
  }
  const formatDateValue = (value) =>
    value
      ? new Date(value).toLocaleDateString("en-IN", {
          day: "2-digit",
          month: "short",
          year: "numeric",
        })
      : "-"
  const formatTimeValue = (value) =>
    value
      ? new Date(value).toLocaleTimeString("en-IN", {
          hour: "2-digit",
          minute: "2-digit",
        })
      : "-"

  const renderCustomerBalanceLabel = (balance, explicitDisplayBalance = null) => {
    const amount = getBalanceDisplayValue(balance, explicitDisplayBalance)
    if (amount > 0) return `Advance Available: Rs. ${formatMoney(amount)}`
    if (amount < 0) return `Pending Amount: -Rs. ${formatMoney(Math.abs(amount))}`
    return "No pending or advance"
  }

  const getStoredBalanceTextClass = (balance, explicitDisplayBalance = null) => {
    const amount = getBalanceDisplayValue(balance, explicitDisplayBalance)
    if (amount < 0) return "text-orange-600"
    if (amount > 0) return "text-emerald-600"
    return "text-muted-foreground"
  }

  const getBalanceDisplayValue = (balance, explicitDisplayBalance = null) => {
    const value = hasNumericBalance(explicitDisplayBalance)
      ? Number(explicitDisplayBalance)
      : -Number(balance || 0)
    return Math.abs(value) < 0.005 ? 0 : value
  }

  const getDisplayBalanceTextClass = (displayBalance) => {
    const amount = Number(displayBalance || 0)
    if (amount > 0) return "text-emerald-600"
    if (amount < 0) return "text-orange-600"
    return "text-muted-foreground"
  }

  const getLedgerEntryDisplayBalance = (entry) =>
    getBalanceDisplayValue(entry?.balance, entry?.display_balance)

  const getLedgerTypeLabel = (type) => {
    const t = String(type || "").toUpperCase()
    if (t === "ADVANCE_IN") return "Advance Added"
    if (t === "PAYMENT" || t === "PAYMENT_IN" || t === "SALE_PAYMENT") return "Payment Received"
    if (t === "ADVANCE_USED") return "Advance Used"
    return String(type || "-").replaceAll("_", " ")
  }

  const getSupplierLedgerTypeLabel = (type) => {
    const t = String(type || "").toUpperCase()
    if (t === "SUPPLIER_BILL") return "Bill Added"
    if (t === "SUPPLIER_PAYMENT" || t === "PAYMENT_OUT") return "Payment Made"
    if (t === "OPENING_BALANCE") return "Opening Balance"
    return String(type || "-").replaceAll("_", " ")
  }

  const handleSupplierBillLookupChange = (value) => {
    setSupplierBillSearch(value)
    const matchedSupplier = suppliers.find((supplier) => getSupplierLookupLabel(supplier) === value)
    setSupplierBillSupplierId(matchedSupplier?.id || "")
  }

  const handleSupplierPaymentLookupChange = (value) => {
    setSupplierPaymentSearch(value)
    const matchedInvoice = pendingSupplierInvoices.find(
      (invoice) => getSupplierInvoiceLookupLabel(invoice) === value
    )
    setSupplierPaymentInvoiceId(matchedInvoice?.id || "")
    setSupplierPaymentAmount(matchedInvoice ? String(Number(matchedInvoice.pending_amount || 0)) : "")
  }

  function matchesSearchText(fields, rawSearch) {
    const query = String(rawSearch || "").trim().toLowerCase()
    if (!query) return true

    return fields.some((field) =>
      String(field || "").toLowerCase().includes(query)
    )
  }

  const getInvoiceItemSummary = (invoice) => {
    const names = (invoice?.items || [])
      .map((item) => item?.product_name)
      .filter(Boolean)

    if (names.length === 0) return "-"
    if (names.length <= 2) return names.join(", ")
    return `${names.slice(0, 2).join(", ")} +${names.length - 2} more`
  }

  const openCustomerDetails = async (customer) => {
    setViewCustomer(customer)
    setCustomerLedger(null)
    setCustomerLedgerLoading(true)
    setCustomerDetailTab("ledger")
    setCustomerLedgerSearch("")
    setCustomerInvoiceSearch("")
    setCustomerInvoiceStatusFilter("all")
    setCustomerItemSearch("")
    setSelectedCustomerInvoice(null)

    try {
      const res = await axios.get(`${API}/accounts/customer-ledger/${customer.id}`, tokenHeaders())
      const ledgerCustomer = res.data?.customer || {}

      setViewCustomer({
        ...customer,
        ...ledgerCustomer,
        email: ledgerCustomer.email || customer.email || "",
        address: ledgerCustomer.address || customer.address || "",
        total_invoices: ledgerCustomer.total_invoices ?? customer.total_invoices ?? 0,
        total_bill: ledgerCustomer.total_bill ?? customer.total_bill ?? 0,
      })
      setCustomerLedger(res.data)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to load customer ledger")
    } finally {
      setCustomerLedgerLoading(false)
    }
  }

  const openSupplierDetails = async (supplier) => {
    setViewSupplier(supplier)
    setSupplierLedger(null)
    setSupplierLedgerLoading(true)
    setSupplierLedgerSearch("")
    setSupplierDetailTab("ledger")

    try {
      const res = await axios.get(`${API}/accounts/supplier-ledger/${supplier.id}`, tokenHeaders())
      setViewSupplier({
        ...supplier,
        ...(res.data?.supplier || {}),
      })
      setSupplierLedger(res.data)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to load supplier ledger")
    } finally {
      setSupplierLedgerLoading(false)
    }
  }

  const openAdvanceDialogForCustomer = (customer) => {
    const pendingAmount = Number(customer?.pending_balance || 0)
    const canAddAdvance = customer?.can_add_advance ?? pendingAmount <= 0

    if (!canAddAdvance || pendingAmount > 0) {
      toast.error("Advance cannot be added while pending balance exists")
      return
    }

    setAdvanceCustomer(customer)
    setAdvanceForm(emptyAdvanceForm)
    setAdvanceDialogOpen(true)
  }

  const submitAdvance = async () => {
    if (!advanceCustomer?.id) return

    if (!advanceForm.amount || Number(advanceForm.amount) <= 0) {
      toast.error("Enter a valid advance amount")
      return
    }

    try {
      setAdvanceSubmitting(true)

      await axios.post(
        `${API}/customers/${advanceCustomer.id}/advance`,
        {
          amount: Number(advanceForm.amount),
          payment_mode: advanceForm.payment_mode,
          reference: advanceForm.reference?.trim() || null,
        },
        tokenHeaders()
      )

      toast.success("Advance added successfully")
      setAdvanceDialogOpen(false)
      setAdvanceCustomer(null)
      setAdvanceForm(emptyAdvanceForm)

      const refreshedCustomers = await fetchCustomers()

      if (viewCustomer?.id === advanceCustomer.id) {
        const refreshedCustomer =
          refreshedCustomers.find((customer) => customer.id === advanceCustomer.id) ||
          { ...viewCustomer, current_balance: Number(viewCustomer.current_balance || 0) - Number(advanceForm.amount || 0) }

        await openCustomerDetails(refreshedCustomer)
      }
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add advance")
    } finally {
      setAdvanceSubmitting(false)
    }
  }

  const customerLedgerEntries = customerLedger?.data || []
  const customerInvoices = customerLedger?.invoices || []
  const customerItemLedger = customerLedger?.item_ledger || []
  const filteredCustomerLedgerEntries = customerLedgerEntries.filter((entry) =>
    matchesSearchText(
      [
        entry?.date,
        entry?.description,
        entry?.type,
        getLedgerTypeLabel(entry?.type),
        entry?.mode,
        entry?.amount,
        entry?.debit,
        entry?.credit,
        entry?.balance,
      ],
      customerLedgerSearch
    )
  )
  const filteredCustomerInvoices = customerInvoices.filter((invoice) => {
    const statusMatches =
      customerInvoiceStatusFilter === "all" ||
      String(invoice?.payment_status || "").toLowerCase() === customerInvoiceStatusFilter

    if (!statusMatches) return false

    const itemFields = (invoice?.items || []).flatMap((item) => [
      item?.product_name,
      item?.sku,
      item?.v_sku,
      item?.variant_name,
      item?.color,
      item?.size,
      item?.variant_info?.v_sku,
      item?.variant_info?.variant_name,
      item?.variant_info?.color,
      item?.variant_info?.size,
    ])

    return matchesSearchText(
      [
        invoice?.invoice_number,
        invoice?.payment_status,
        invoice?.payment_mode,
        invoice?.created_at,
        invoice?.total,
        invoice?.paid_amount,
        invoice?.balance_amount,
        invoice?.advance_used,
        ...itemFields,
      ],
      customerInvoiceSearch
    )
  })
  const filteredCustomerItemLedger = customerItemLedger.filter((item) =>
    matchesSearchText(
      [
        item?.invoice_number,
        item?.product_name,
        item?.sku,
        item?.variant_name,
        item?.color,
        item?.size,
        item?.payment_status,
        item?.created_at,
        item?.quantity,
        item?.price,
        item?.total,
      ],
      customerItemSearch
    )
  )
  const supplierLedgerEntries = [...(supplierLedger?.data || [])].sort((leftEntry, rightEntry) => {
    const leftValue = String(leftEntry?.created_at || leftEntry?.date || "")
    const rightValue = String(rightEntry?.created_at || rightEntry?.date || "")
    return rightValue.localeCompare(leftValue)
  })
  const supplierLedgerInvoices = supplierLedger?.invoices || []
  const filteredSupplierLedgerEntries = supplierLedgerEntries.filter((entry) =>
    matchesSearchText(
      [
        entry?.date,
        entry?.created_at,
        entry?.description,
        entry?.type,
        getSupplierLedgerTypeLabel(entry?.type),
        entry?.mode,
        entry?.amount,
        entry?.debit,
        entry?.credit,
        entry?.balance,
        entry?.bill_number,
        entry?.status,
      ],
      supplierLedgerSearch
    )
  )
  const filteredSupplierInvoices = supplierLedgerInvoices.filter((invoice) =>
    matchesSearchText(
      [
        invoice?.id,
        invoice?.bill_number,
        invoice?.supplier_name,
        invoice?.invoice_date,
        invoice?.status,
        invoice?.total_amount,
        invoice?.paid_amount,
        invoice?.pending_amount,
      ],
      supplierLedgerSearch
    )
  )
  const viewCustomerBalance =
    customerLedger?.customer?.current_balance ??
    customerLedger?.customer?.stored_current_balance ??
    viewCustomer?.current_balance
  const viewCustomerDisplayBalance = getBalanceDisplayValue(
    viewCustomerBalance,
    customerLedger?.customer?.display_balance
  )
  const viewCustomerPendingBalance = Number(
    customerLedger?.customer?.pending_balance ??
    viewCustomer?.pending_balance ??
    0
  )
  const viewCustomerAdvanceBalance = Number(
    customerLedger?.customer?.advance_balance ??
    viewCustomer?.advance_balance ??
    0
  )
  const viewCustomerCanAddAdvance =
    customerLedger?.customer?.can_add_advance ??
    (viewCustomer?.can_add_advance ?? (Number(viewCustomer?.pending_balance || 0) <= 0))
  const viewCustomerInvoiceCount = Number(
    customerLedger?.customer?.total_invoices ??
    viewCustomer?.total_invoices ??
    customerInvoices.length
  )
  const viewCustomerTotalBill = Number(
    customerLedger?.customer?.total_bill ??
    viewCustomer?.total_bill ??
    0
  )
  const viewSupplierDetails = supplierLedger?.supplier || viewSupplier || null
  const viewSupplierOutstanding = Number(viewSupplierDetails?.pending_amount || 0)
  const viewSupplierComputedBalance = Number(
    viewSupplierDetails?.computed_balance ??
    viewSupplierDetails?.current_balance ??
    0
  )

  const getInvoiceStatusClass = (status) => {
    const value = String(status || "").toLowerCase()
    if (value === "paid") return "bg-emerald-100 text-emerald-700"
    if (value === "partial") return "bg-amber-100 text-amber-700"
    if (value === "cancelled") return "bg-rose-100 text-rose-700"
    return "bg-orange-100 text-orange-700"
  }

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto max-w-7xl space-y-4 p-4 md:p-6 lg:p-8">
        <div className="flex flex-col gap-3 rounded-[30px] border border-white/10 bg-slate-950/70 p-3 shadow-[0_22px_60px_rgba(15,23,42,0.26)] lg:flex-row lg:items-center">
          <Dialog
            open={openCustomer}
            onOpenChange={(next) => {
              setOpenCustomer(next)
              if (!next) resetCustomerForm()
            }}
          >
            {activeTab === "customers" && (
              <DialogTrigger asChild>
                <Button
                  size="lg"
                  className="h-14 w-full rounded-[22px] bg-indigo-500 px-5 text-white shadow-[0_18px_40px_rgba(99,102,241,0.35)] hover:bg-indigo-400 sm:w-auto"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  {editingCustomer ? "Edit Customer" : "Add Customer"}
                </Button>
              </DialogTrigger>
            )}

            <DialogContent className="max-w-[95vw] rounded-3xl sm:max-w-[500px]">
              <DialogHeader>
                <DialogTitle>{editingCustomer ? "Edit Customer" : "Add New Customer"}</DialogTitle>
              </DialogHeader>

              <form onSubmit={submitCustomer} className="space-y-4">
                <div className="space-y-2">
                  <Label htmlFor="name">Full Name</Label>
                  <Input
                    id="name"
                    placeholder="Enter customer name"
                    value={customerForm.name}
                    onChange={(e) => setCustomerForm({ ...customerForm, name: e.target.value })}
                    required
                    className="h-12 rounded-2xl"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    type="email"
                    placeholder="customer@example.com"
                    value={customerForm.email}
                    onChange={(e) => setCustomerForm({ ...customerForm, email: e.target.value })}
                    className="h-12 rounded-2xl"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="phone">Phone Number</Label>
                  <Input
                    id="phone"
                    placeholder="+91 98765 43210"
                    value={customerForm.phone}
                    onChange={(e) => setCustomerForm({ ...customerForm, phone: e.target.value })}
                    className="h-12 rounded-2xl"
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="address">Address</Label>
                  <Input
                    id="address"
                    placeholder="Street address, city, state"
                    value={customerForm.address}
                    onChange={(e) => setCustomerForm({ ...customerForm, address: e.target.value })}
                    className="h-12 rounded-2xl"
                  />
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                  {editingCustomer && (
                    <Button type="button" variant="outline" className="h-12 rounded-2xl" onClick={resetCustomerForm}>
                      Cancel Edit
                    </Button>
                  )}
                  <Button type="submit" className="h-12 rounded-2xl">
                    {editingCustomer ? "Update Customer" : "Add Customer"}
                  </Button>
                </div>
              </form>
            </DialogContent>
          </Dialog>

          {activeTab === "suppliers" && (
            <Dialog open={openSupplier} onOpenChange={setOpenSupplier}>
              <DialogTrigger asChild>
                <Button
                  size="lg"
                  variant="secondary"
                  className="h-14 w-full rounded-[22px] border border-white/10 bg-slate-900 px-5 text-white hover:bg-slate-800 sm:w-auto"
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Add Supplier
                </Button>
              </DialogTrigger>

              <DialogContent className="max-w-[95vw] rounded-3xl sm:max-w-[500px]">
                <DialogHeader>
                  <DialogTitle>Add New Supplier</DialogTitle>
                </DialogHeader>

                <form onSubmit={addSupplier} className="space-y-4">
                  <div className="space-y-2">
                    <Label htmlFor="supplier-name">Supplier Name</Label>
                    <Input
                      id="supplier-name"
                      placeholder="Enter supplier name"
                      value={supplierForm.name}
                      onChange={(e) => setSupplierForm({ ...supplierForm, name: e.target.value })}
                      required
                      className="h-12 rounded-2xl"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="supplier-phone">Phone Number</Label>
                    <Input
                      id="supplier-phone"
                      placeholder="+91 98765 43210"
                      value={supplierForm.phone}
                      onChange={(e) => setSupplierForm({ ...supplierForm, phone: e.target.value })}
                      className="h-12 rounded-2xl"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="supplier-email">Email</Label>
                    <Input
                      id="supplier-email"
                      type="email"
                      placeholder="supplier@example.com"
                      value={supplierForm.email}
                      onChange={(e) => setSupplierForm({ ...supplierForm, email: e.target.value })}
                      className="h-12 rounded-2xl"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="supplier-address">Address</Label>
                    <Input
                      id="supplier-address"
                      placeholder="Street address, city, state"
                      value={supplierForm.address}
                      onChange={(e) => setSupplierForm({ ...supplierForm, address: e.target.value })}
                      className="h-12 rounded-2xl"
                    />
                  </div>

                  <Button type="submit" className="h-12 w-full rounded-2xl">
                    Add Supplier
                  </Button>
                </form>
              </DialogContent>
            </Dialog>
          )}

          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              placeholder={activeTab === "customers" ? "Search customer..." : "Search supplier..."}
              className="h-14 rounded-full border-white/10 bg-slate-950/50 pl-11 pr-4 text-sm text-white placeholder:text-slate-400"
              value={activeTab === "customers" ? searchTerm : supplierSearch}
              onChange={(e) =>
                activeTab === "customers"
                  ? setSearchTerm(e.target.value)
                  : setSupplierSearch(e.target.value)
              }
            />
          </div>
          <div className="flex gap-1 rounded-full border border-white/10 bg-slate-900/90 p-1.5 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] sm:w-auto">
            <Button
              onClick={() => setActiveTab("customers")}
              variant="ghost"
              className={`h-11 gap-2 rounded-full px-4 text-xs font-semibold ${
                activeTab === "customers"
                  ? "bg-indigo-500 text-white shadow-[0_12px_28px_rgba(99,102,241,0.35)] hover:bg-indigo-400"
                  : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
            >
              <User className="h-3 w-3" />
              <span>Customers</span>
            </Button>

            <Button
              onClick={() => setActiveTab("suppliers")}
              variant="ghost"
              className={`h-11 gap-2 rounded-full px-4 text-xs font-semibold ${
                activeTab === "suppliers"
                  ? "bg-indigo-500 text-white shadow-[0_12px_28px_rgba(99,102,241,0.35)] hover:bg-indigo-400"
                  : "text-slate-300 hover:bg-white/5 hover:text-white"
              }`}
            >
              <Building2 className="h-3 w-3" />
              <span>Suppliers</span>
            </Button>
          </div>
        </div>

        {activeTab === "customers" && (
          <div className="space-y-3">
            <div className="flex flex-col gap-2 rounded-2xl border bg-muted/20 p-3 sm:flex-row sm:items-center sm:justify-between">
              <div className="text-sm text-muted-foreground">
                Showing {paginatedCustomers.length} of {sortedCustomers.length} filtered customers
              </div>
              <div className="flex items-center gap-2">
                <Label className="text-xs text-muted-foreground">Sort</Label>
                <Select value={customerSort} onValueChange={setCustomerSort}>
                  <SelectTrigger className="h-10 w-40 rounded-xl">
                    <SelectValue placeholder="Sort customers" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="a-z">A-Z</SelectItem>
                    <SelectItem value="z-a">Z-A</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            <Card className="border-border/60 shadow-sm">
              <CardContent className="p-2">
                <div className="space-y-1.5 md:hidden">
                  {sortedCustomers.length === 0 && (
                    <div className="rounded-3xl border border-dashed bg-muted/10 px-6 py-12 text-center text-sm text-muted-foreground">
                      No customers found
                    </div>
                  )}

                  {paginatedCustomers.map((customer) => (
                    <Card key={customer.id} className="rounded-xl border shadow-sm">
                      <CardContent className="space-y-2 p-2.5">
                        <div
                          className="cursor-pointer"
                          onClick={() => openCustomerDetails(customer)}
                        >
                          <div className="flex items-start gap-2">
                            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-primary/10 text-primary">
                              <User className="h-4 w-4" />
                            </div>
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center justify-between gap-2">
                                <p className="truncate text-sm font-semibold">{customer.name}</p>
                                <p className="text-xs font-semibold text-green-600">
                                ₹{Number(customer.total_bill || 0).toLocaleString("en-IN")}
                                </p>
                              </div>
                              <div className="mt-1 flex items-center gap-1.5 text-[11px] text-muted-foreground">
                                <Phone className="h-3 w-3" />
                                <span className="truncate">{customer.phone || "-"}</span>
                              </div>
                              <p className="mt-1 text-[11px] text-muted-foreground">
                                {Number(customer.total_invoices || 0)} invoice{Number(customer.total_invoices || 0) === 1 ? "" : "s"}
                              </p>
                              <p className={`mt-1 text-[11px] ${getStoredBalanceTextClass(customer.current_balance, customer.display_balance)}`}>
                                {renderCustomerBalanceLabel(customer.current_balance, customer.display_balance)}
                              </p>
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-5 gap-1 border-t pt-2">
                          <Button size="sm" variant="outline" className="h-7 rounded-md px-1.5 text-[10px]" onClick={() => openCustomerDetails(customer)}>
                            View
                          </Button>
                          <Button size="sm" variant="outline" className="h-7 rounded-md px-1.5 text-[10px]" onClick={() => handleCreateInvoice(customer)}>
                            Invoice
                          </Button>
                          <Button size="sm" variant="outline" className="h-7 rounded-md px-1.5 text-[10px]" onClick={() => openAdvanceDialogForCustomer(customer)}>
                            Adv
                          </Button>
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-7 rounded-md px-1.5 text-[10px]"
                            onClick={() => {
                              setEditingCustomer(customer)
                              setCustomerForm({
                                name: customer.name || "",
                                email: customer.email || "",
                                phone: customer.phone || "",
                                address: customer.address || "",
                              })
                              setOpenCustomer(true)
                            }}
                          >
                            Edit
                          </Button>
                          <Button size="sm" variant="destructive" className="h-7 rounded-md px-1.5 text-[10px]" onClick={() => deleteCustomer(customer)}>
                            Delete
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))}
                </div>

                <div className="hidden overflow-x-auto md:block">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/50">
                        <TableHead>#</TableHead>
                        <TableHead>Name</TableHead>
                        <TableHead>Email</TableHead>
                        <TableHead>Phone</TableHead>
                        <TableHead>Invoices</TableHead>
                        <TableHead className="text-right">Total Billing</TableHead>
                        <TableHead>Balance</TableHead>
                        <TableHead>Address</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedCustomers.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan="9" className="py-8 text-center text-muted-foreground">
                            No customers found
                          </TableCell>
                        </TableRow>
                      ) : (
                        paginatedCustomers.map((customer, index) => (
                          <TableRow
                            key={customer.id}
                            className="cursor-pointer hover:bg-muted/40"
                            onClick={() => openCustomerDetails(customer)}
                          >
                            <TableCell>{(safeCustomerPage - 1) * CUSTOMERS_PER_PAGE + index + 1}</TableCell>
                            <TableCell className="font-medium">{customer.name}</TableCell>
                            <TableCell>{customer.email || "-"}</TableCell>
                            <TableCell>{customer.phone || "-"}</TableCell>
                            <TableCell>{Number(customer.total_invoices || 0)}</TableCell>
                            <TableCell className="text-right font-semibold text-green-600">
                              ₹{Number(customer.total_bill || 0).toLocaleString("en-IN")}
                            </TableCell>
                            <TableCell className={getStoredBalanceTextClass(customer.current_balance, customer.display_balance)}>
                              {renderCustomerBalanceLabel(customer.current_balance, customer.display_balance)}
                            </TableCell>
                            <TableCell className="max-w-xs truncate">{customer.address || "-"}</TableCell>
                            <TableCell className="text-right" onClick={(e) => e.stopPropagation()}>
                              <div className="flex justify-end gap-2">
                                <Button size="sm" variant="outline" onClick={() => handleCreateInvoice(customer)}>
                                  <FileText className="mr-2 h-4 w-4" />
                                  Invoice
                                </Button>
                                <Button size="sm" variant="outline" onClick={() => openAdvanceDialogForCustomer(customer)}>
                                  <Wallet className="mr-2 h-4 w-4" />
                                  Advance
                                </Button>
                                <Button
                                  size="sm"
                                  variant="outline"
                                  onClick={() => {
                                    setEditingCustomer(customer)
                                    setCustomerForm({
                                      name: customer.name || "",
                                      email: customer.email || "",
                                      phone: customer.phone || "",
                                      address: customer.address || "",
                                    })
                                    setOpenCustomer(true)
                                  }}
                                >
                                  <Edit className="mr-2 h-4 w-4" />
                                  Edit
                                </Button>
                                <Button size="sm" variant="destructive" onClick={() => deleteCustomer(customer)}>
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>

            <div className="flex flex-col gap-2 px-1 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
              <span>
                Page {safeCustomerPage} of {customerTotalPages} • {sortedCustomers.length} filtered / {customers.length} total
              </span>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-xl"
                  disabled={safeCustomerPage === 1}
                  onClick={() => setCustomerPage((prev) => Math.max(prev - 1, 1))}
                >
                  Previous
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="rounded-xl"
                  disabled={safeCustomerPage === customerTotalPages}
                  onClick={() => setCustomerPage((prev) => Math.min(prev + 1, customerTotalPages))}
                >
                  Next
                </Button>
              </div>
            </div>
          </div>
        )}

        <Dialog
          open={!!viewCustomer}
          onOpenChange={(open) => {
            if (!open) {
              setViewCustomer(null)
              setCustomerLedger(null)
              setCustomerDetailTab("ledger")
              setCustomerLedgerSearch("")
              setCustomerInvoiceSearch("")
              setCustomerInvoiceStatusFilter("all")
              setCustomerItemSearch("")
              setSelectedCustomerInvoice(null)
            }
          }}
        >
          <DialogContent className="max-h-[90vh] w-[calc(100vw-1rem)] max-w-[calc(100vw-1rem)] overflow-x-hidden overflow-y-auto rounded-3xl p-4 sm:max-w-5xl sm:p-6">
            <DialogHeader>
              <DialogTitle>Customer Details</DialogTitle>
            </DialogHeader>
            {viewCustomer && (
              <div className="min-w-0 space-y-5 text-sm">
                <div className="flex flex-col gap-3 overflow-hidden rounded-2xl border bg-muted/20 p-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-lg font-semibold">{viewCustomer.name}</p>
                    <div className="mt-2 space-y-1 text-muted-foreground">
                      <p>Phone: {viewCustomer.phone || "-"}</p>
                      <p>Email: {viewCustomer.email || "-"}</p>
                      <p>Address: {viewCustomer.address || "-"}</p>
                      <p className={`${getStoredBalanceTextClass(viewCustomerBalance, customerLedger?.customer?.display_balance ?? viewCustomer?.display_balance)} font-medium`}>
                        {renderCustomerBalanceLabel(viewCustomerBalance, customerLedger?.customer?.display_balance ?? viewCustomer?.display_balance)}
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 md:min-w-[26rem]">
                    <div className="rounded-2xl bg-background p-3 shadow-sm">
                      <p className="text-xs text-muted-foreground">Invoices</p>
                      <p className="text-base font-semibold">{viewCustomerInvoiceCount}</p>
                    </div>
                    <div className="rounded-2xl bg-background p-3 shadow-sm">
                      <p className="text-xs text-muted-foreground">Total Sale</p>
                      <p className="text-base font-semibold text-green-600">Rs. {formatMoney(viewCustomerTotalBill)}</p>
                    </div>
                    <div className="rounded-2xl bg-background p-3 shadow-sm">
                      <p className="text-xs text-muted-foreground">Total Balance</p>
                      <p
                        className={`text-base font-semibold ${
                          getDisplayBalanceTextClass(viewCustomerDisplayBalance)
                        }`}
                      >
                        Rs. {formatMoney(viewCustomerDisplayBalance)}
                      </p>
                    </div>
                    <div className="rounded-2xl bg-background p-3 shadow-sm">
                      <p className="text-xs text-muted-foreground">Pending / Advance</p>
                      <p className="text-base font-semibold">
                        {viewCustomerPendingBalance > 0
                          ? `Pending: Rs. ${formatMoney(viewCustomerPendingBalance)}`
                          : viewCustomerAdvanceBalance > 0
                            ? `Advance: Rs. ${formatMoney(viewCustomerAdvanceBalance)}`
                            : "Clear"}
                      </p>
                    </div>
                  </div>
                </div>

                {customerLedgerLoading ? (
                  <div className="rounded-2xl border border-dashed p-8 text-center text-muted-foreground">
                    Loading customer ledger...
                  </div>
                ) : (
                  <>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        variant={customerDetailTab === "ledger" ? "default" : "outline"}
                        className="rounded-2xl"
                        onClick={() => setCustomerDetailTab("ledger")}
                      >
                        Ledger
                      </Button>
                      <Button
                        type="button"
                        variant={customerDetailTab === "invoices" ? "default" : "outline"}
                        className="rounded-2xl"
                        onClick={() => setCustomerDetailTab("invoices")}
                      >
                        Invoices
                      </Button>
                      <Button
                        type="button"
                        variant={customerDetailTab === "items" ? "default" : "outline"}
                        className="rounded-2xl"
                        onClick={() => setCustomerDetailTab("items")}
                      >
                        Item Ledger
                      </Button>
                    </div>

                    {customerDetailTab === "ledger" && (
                      <div className="space-y-3">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                          <h3 className="text-base font-semibold">Ledger Entries</h3>
                          <div className="relative w-full sm:max-w-sm">
                            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            <Input
                              placeholder="Search ledger..."
                              value={customerLedgerSearch}
                              onChange={(e) => setCustomerLedgerSearch(e.target.value)}
                              className="h-10 rounded-2xl pl-9"
                            />
                          </div>
                        </div>
                        <div className="max-h-[24rem] overflow-y-auto rounded-2xl border p-3">
                          {filteredCustomerLedgerEntries.length === 0 ? (
                            <p className="text-sm text-muted-foreground">No ledger entries found</p>
                          ) : (
                            <div className="overflow-x-auto rounded-xl border">
                              <Table>
                                <TableHeader>
                                  <TableRow className="bg-muted/30">
                                    <TableHead>Date</TableHead>
                                    <TableHead>Type</TableHead>
                                    <TableHead>Mode</TableHead>
                                    <TableHead>Description</TableHead>
                                    <TableHead className="text-right">Amount</TableHead>
                                    <TableHead className="text-right">Debit</TableHead>
                                    <TableHead className="text-right">Credit</TableHead>
                                    <TableHead className="text-right">Balance</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {filteredCustomerLedgerEntries.map((entry, index) => (
                                    <TableRow key={entry.id || `${entry.date || "entry"}-${index}`}>
                                      <TableCell>{entry.date ? new Date(entry.date).toLocaleDateString("en-IN") : "-"}</TableCell>
                                      <TableCell>{getLedgerTypeLabel(entry.type)}</TableCell>
                                      <TableCell className="uppercase text-muted-foreground">{entry.mode || "-"}</TableCell>
                                      <TableCell className="max-w-sm">{entry.description || "-"}</TableCell>
                                      <TableCell className="text-right font-medium">
                                        {Number(entry.amount || 0) > 0 ? `Rs. ${formatMoney(entry.amount)}` : "-"}
                                      </TableCell>
                                      <TableCell className="text-right text-orange-600">Rs. {formatMoney(entry.debit)}</TableCell>
                                      <TableCell className="text-right text-green-600">Rs. {formatMoney(entry.credit)}</TableCell>
                                      <TableCell className={`text-right font-semibold ${getDisplayBalanceTextClass(getLedgerEntryDisplayBalance(entry))}`}>
                                        Rs. {formatMoney(getLedgerEntryDisplayBalance(entry))}
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {customerDetailTab === "invoices" && (
                      <div className="space-y-3">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                          <h3 className="text-base font-semibold">Customer Invoices</h3>
                          <p className="text-xs text-muted-foreground">
                            {filteredCustomerInvoices.length} of {customerInvoices.length} invoice(s)
                          </p>
                        </div>
                        <div className="flex flex-col gap-2 sm:flex-row">
                          <div className="relative flex-1">
                            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                            <Input
                              placeholder="Search invoice number, status, item or SKU..."
                              value={customerInvoiceSearch}
                              onChange={(e) => setCustomerInvoiceSearch(e.target.value)}
                              className="h-10 rounded-2xl pl-9"
                            />
                          </div>
                          <Select value={customerInvoiceStatusFilter} onValueChange={setCustomerInvoiceStatusFilter}>
                            <SelectTrigger className="h-10 rounded-2xl sm:w-44">
                              <SelectValue placeholder="Filter status" />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="all">All Status</SelectItem>
                              <SelectItem value="pending">Pending</SelectItem>
                              <SelectItem value="partial">Partial</SelectItem>
                              <SelectItem value="paid">Paid</SelectItem>
                              <SelectItem value="cancelled">Cancelled</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="max-h-[28rem] overflow-y-auto rounded-2xl border p-3">
                          {filteredCustomerInvoices.length === 0 ? (
                            <p className="text-sm text-muted-foreground">No invoices found for this customer</p>
                          ) : (
                            <div className="overflow-x-auto rounded-xl border">
                              <Table>
                                <TableHeader>
                                  <TableRow className="bg-muted/30">
                                    <TableHead>Invoice</TableHead>
                                    <TableHead>Date</TableHead>
                                    <TableHead>Status</TableHead>
                                    <TableHead>Items</TableHead>
                                    <TableHead className="text-right">Total</TableHead>
                                    <TableHead className="text-right">Paid</TableHead>
                                    <TableHead className="text-right">Pending</TableHead>
                                    <TableHead className="text-right">Advance</TableHead>
                                    <TableHead className="text-right">Actions</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {filteredCustomerInvoices.map((invoice) => (
                                    <TableRow key={invoice.id}>
                                      <TableCell>
                                        <div>
                                          <p className="font-medium">{invoice.invoice_number || "-"}</p>
                                          <p className="text-xs text-muted-foreground capitalize">
                                            {invoice.payment_mode || "-"}
                                          </p>
                                        </div>
                                      </TableCell>
                                      <TableCell>{invoice.created_at ? new Date(invoice.created_at).toLocaleDateString("en-IN") : "-"}</TableCell>
                                      <TableCell>
                                        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${getInvoiceStatusClass(invoice.payment_status)}`}>
                                          {String(invoice.payment_status || "pending").toUpperCase()}
                                        </span>
                                      </TableCell>
                                      <TableCell className="max-w-xs">
                                        <div className="truncate" title={getInvoiceItemSummary(invoice)}>
                                          {getInvoiceItemSummary(invoice)}
                                        </div>
                                      </TableCell>
                                      <TableCell className="text-right font-semibold">Rs. {formatMoney(invoice.total)}</TableCell>
                                      <TableCell className="text-right">Rs. {formatMoney(invoice.paid_amount)}</TableCell>
                                      <TableCell className="text-right text-orange-600">Rs. {formatMoney(invoice.balance_amount)}</TableCell>
                                      <TableCell className="text-right text-emerald-600">Rs. {formatMoney(invoice.advance_used)}</TableCell>
                                      <TableCell className="text-right">
                                        <Button
                                          size="sm"
                                          variant="outline"
                                          className="rounded-xl"
                                          onClick={() => setSelectedCustomerInvoice(invoice)}
                                        >
                                          View
                                        </Button>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    {customerDetailTab === "items" && (
                      <div className="space-y-3">
                        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                          <h3 className="text-base font-semibold">Item Ledger</h3>
                          <p className="text-xs text-muted-foreground">
                            {filteredCustomerItemLedger.length} of {customerItemLedger.length} item row(s)
                          </p>
                        </div>
                        <div className="relative">
                          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                          <Input
                            placeholder="Search item, SKU, invoice or status..."
                            value={customerItemSearch}
                            onChange={(e) => setCustomerItemSearch(e.target.value)}
                            className="h-10 rounded-2xl pl-9"
                          />
                        </div>
                        <div className="max-h-[28rem] overflow-y-auto rounded-2xl border p-3">
                          {filteredCustomerItemLedger.length === 0 ? (
                            <p className="text-sm text-muted-foreground">No invoice items found for this customer</p>
                          ) : (
                            <div className="overflow-x-auto rounded-xl border">
                              <Table>
                                <TableHeader>
                                  <TableRow className="bg-muted/30">
                                    <TableHead>Date</TableHead>
                                    <TableHead>Invoice</TableHead>
                                    <TableHead>Item</TableHead>
                                    <TableHead>SKU</TableHead>
                                    <TableHead className="text-right">Qty</TableHead>
                                    <TableHead className="text-right">Price</TableHead>
                                    <TableHead className="text-right">Total</TableHead>
                                    <TableHead>Status</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {filteredCustomerItemLedger.map((item) => (
                                    <TableRow key={item.id}>
                                      <TableCell>{item.created_at ? new Date(item.created_at).toLocaleDateString("en-IN") : "-"}</TableCell>
                                      <TableCell className="font-medium">{item.invoice_number || "-"}</TableCell>
                                      <TableCell>
                                        <div>
                                          <p className="font-medium">{item.product_name || "-"}</p>
                                          {(item.variant_name || item.color || item.size) && (
                                            <p className="text-xs text-muted-foreground">
                                              {[item.variant_name, item.color, item.size].filter(Boolean).join(" • ")}
                                            </p>
                                          )}
                                        </div>
                                      </TableCell>
                                      <TableCell className="font-mono text-xs">{item.sku || "-"}</TableCell>
                                      <TableCell className="text-right">{Number(item.quantity || 0)}</TableCell>
                                      <TableCell className="text-right">Rs. {formatMoney(item.price)}</TableCell>
                                      <TableCell className="text-right font-semibold">Rs. {formatMoney(item.total)}</TableCell>
                                      <TableCell>
                                        <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${getInvoiceStatusClass(item.payment_status)}`}>
                                          {String(item.payment_status || "pending").toUpperCase()}
                                        </span>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          )}
                        </div>
                      </div>
                    )}

                    <div className="grid grid-cols-1 gap-2 pt-2 md:grid-cols-3">
                      <Button
                        variant="outline"
                        className="rounded-2xl"
                        onClick={() => {
                          setViewCustomer(null)
                          setCustomerLedger(null)
                          setCustomerDetailTab("ledger")
                          setCustomerLedgerSearch("")
                          setCustomerInvoiceSearch("")
                          setCustomerInvoiceStatusFilter("all")
                          setCustomerItemSearch("")
                          setSelectedCustomerInvoice(null)
                        }}
                      >
                        Close
                      </Button>
                      <Button
                        variant="outline"
                        className="rounded-2xl"
                        onClick={() => openAdvanceDialogForCustomer(viewCustomer)}
                        disabled={!viewCustomerCanAddAdvance}
                      >
                        Add Advance
                      </Button>
                      <Button className="rounded-2xl" onClick={() => handleCreateInvoice(viewCustomer)}>
                        Create Invoice
                      </Button>
                    </div>
                  </>
                )}
              </div>
            )}
          </DialogContent>
        </Dialog>

        <Dialog open={!!selectedCustomerInvoice} onOpenChange={(open) => !open && setSelectedCustomerInvoice(null)}>
          <DialogContent className="max-h-[90vh] max-w-[95vw] overflow-y-auto rounded-3xl sm:max-w-4xl">
            <DialogHeader>
              <DialogTitle>Invoice View</DialogTitle>
            </DialogHeader>
            {selectedCustomerInvoice && (
              <div className="space-y-4 text-sm">
                <div className="flex flex-col gap-3 rounded-2xl border bg-muted/20 p-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-lg font-semibold">{selectedCustomerInvoice.invoice_number || "-"}</p>
                    <p className="mt-1 text-muted-foreground">
                      {selectedCustomerInvoice.created_at
                        ? new Date(selectedCustomerInvoice.created_at).toLocaleString("en-IN")
                        : "-"}
                    </p>
                    <p className="mt-1 text-muted-foreground capitalize">
                      Payment Mode: {selectedCustomerInvoice.payment_mode || "-"}
                    </p>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <span className={`rounded-full px-3 py-1 text-xs font-semibold ${getInvoiceStatusClass(selectedCustomerInvoice.payment_status)}`}>
                      {String(selectedCustomerInvoice.payment_status || "pending").toUpperCase()}
                    </span>
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
                  <div className="rounded-2xl border bg-background p-3">
                    <p className="text-xs text-muted-foreground">Total</p>
                    <p className="text-base font-semibold">Rs. {formatMoney(selectedCustomerInvoice.total)}</p>
                  </div>
                  <div className="rounded-2xl border bg-background p-3">
                    <p className="text-xs text-muted-foreground">Paid</p>
                    <p className="text-base font-semibold text-emerald-600">Rs. {formatMoney(selectedCustomerInvoice.paid_amount)}</p>
                  </div>
                  <div className="rounded-2xl border bg-background p-3">
                    <p className="text-xs text-muted-foreground">Pending</p>
                    <p className="text-base font-semibold text-orange-600">Rs. {formatMoney(selectedCustomerInvoice.balance_amount)}</p>
                  </div>
                  <div className="rounded-2xl border bg-background p-3">
                    <p className="text-xs text-muted-foreground">Advance Used</p>
                    <p className="text-base font-semibold text-emerald-600">Rs. {formatMoney(selectedCustomerInvoice.advance_used)}</p>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-2xl border">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/30">
                        <TableHead>Item</TableHead>
                        <TableHead>SKU</TableHead>
                        <TableHead>Variant</TableHead>
                        <TableHead className="text-right">Qty</TableHead>
                        <TableHead className="text-right">Price</TableHead>
                        <TableHead className="text-right">Total</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {(selectedCustomerInvoice.items || []).length === 0 ? (
                        <TableRow>
                          <TableCell colSpan="6" className="py-6 text-center text-muted-foreground">
                            No invoice items found
                          </TableCell>
                        </TableRow>
                      ) : (
                        (selectedCustomerInvoice.items || []).map((item, index) => (
                          <TableRow key={`${selectedCustomerInvoice.id}-${index}`}>
                            <TableCell className="font-medium">{item.product_name || "-"}</TableCell>
                            <TableCell className="font-mono text-xs">
                              {item.v_sku || item.sku || item.variant_info?.v_sku || "-"}
                            </TableCell>
                            <TableCell>
                              {[item.variant_name, item.color, item.size].filter(Boolean).join(" • ") || "-"}
                            </TableCell>
                            <TableCell className="text-right">{Number(item.quantity || 0)}</TableCell>
                            <TableCell className="text-right">Rs. {formatMoney(item.price)}</TableCell>
                            <TableCell className="text-right font-semibold">Rs. {formatMoney(item.total)}</TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </div>

                <div className="flex justify-end">
                  <Button variant="outline" className="rounded-2xl" onClick={() => setSelectedCustomerInvoice(null)}>
                    Close
                  </Button>
                </div>
              </div>
            )}
          </DialogContent>
        </Dialog>

        <Dialog
          open={advanceDialogOpen}
          onOpenChange={(open) => {
            setAdvanceDialogOpen(open)
            if (!open) {
              setAdvanceCustomer(null)
              setAdvanceForm(emptyAdvanceForm)
            }
          }}
        >
          <DialogContent className="max-w-[95vw] rounded-3xl sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Add Customer Advance</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div className="rounded-2xl border bg-muted/20 p-3 text-sm">
                <p className="font-semibold">{advanceCustomer?.name || "-"}</p>
                <p className="text-muted-foreground">
                  {renderCustomerBalanceLabel(
                    advanceCustomer?.current_balance,
                    advanceCustomer?.display_balance
                  )}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Amount</Label>
                <Input
                  type="number"
                  min="0"
                  value={advanceForm.amount}
                  onChange={(e) => setAdvanceForm({ ...advanceForm, amount: e.target.value })}
                  placeholder="Enter advance amount"
                />
              </div>

              <div className="space-y-2">
                <Label>Payment Mode</Label>
                <Select
                  value={advanceForm.payment_mode}
                  onValueChange={(value) => setAdvanceForm({ ...advanceForm, payment_mode: value })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select payment mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cash">Cash</SelectItem>
                    <SelectItem value="upi">UPI</SelectItem>
                    <SelectItem value="bank">Bank</SelectItem>
                    <SelectItem value="cheque">Cheque</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Reference</Label>
                <Input
                  value={advanceForm.reference}
                  onChange={(e) => setAdvanceForm({ ...advanceForm, reference: e.target.value })}
                  placeholder="Optional note / reference"
                />
              </div>

              <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                <Button
                  variant="outline"
                  onClick={() => {
                    setAdvanceDialogOpen(false)
                    setAdvanceCustomer(null)
                    setAdvanceForm(emptyAdvanceForm)
                  }}
                >
                  Cancel
                </Button>
                <Button onClick={submitAdvance} disabled={advanceSubmitting}>
                  {advanceSubmitting ? "Saving..." : "Add Advance"}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {activeTab === "suppliers" && (
          <div className="space-y-4">
            <div className="grid max-w-xl grid-cols-2 gap-3">
              <Card className="border-border/60 shadow-sm">
                <CardContent className="flex items-center justify-between p-3">
                  <div>
                    <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Suppliers</p>
                    <p className="mt-1.5 text-xl font-semibold">{supplierOverview.count}</p>
                  </div>
                  <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-slate-900 text-white">
                    <Building2 className="h-4 w-4" />
                  </div>
                </CardContent>
              </Card>
              <Card className="border-border/60 shadow-sm">
                <CardContent className="p-3">
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Total Bills</p>
                  <p className="mt-1.5 text-lg font-semibold">Rs. {formatMoney(supplierOverview.totalBill)}</p>
                </CardContent>
              </Card>
              <Card className="border-border/60 shadow-sm">
                <CardContent className="p-3">
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Paid</p>
                  <p className="mt-1.5 text-lg font-semibold text-emerald-600">Rs. {formatMoney(supplierOverview.paid)}</p>
                </CardContent>
              </Card>
              <Card className="border-border/60 shadow-sm">
                <CardContent className="p-3">
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Outstanding</p>
                  <p className="mt-1.5 text-lg font-semibold text-rose-600">Rs. {formatMoney(supplierOverview.pending)}</p>
                </CardContent>
              </Card>
            </div>

            <Dialog
              open={supplierBillDialogOpen}
              onOpenChange={(open) => {
                if (!open) {
                  resetSupplierBillDialog()
                  return
                }

                setSupplierBillDialogOpen(true)
              }}
            >
              <DialogContent className="max-h-[90vh] w-[calc(100vw-1rem)] max-w-[calc(100vw-1rem)] overflow-y-auto rounded-3xl sm:max-w-lg">
                <DialogHeader>
                  <DialogTitle>Add Supplier Bill</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="rounded-2xl border bg-muted/20 p-4">
                    <p className="text-sm font-medium">Record a new supplier bill</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Search the supplier, choose the date, and we will update the supplier balance and ledger together.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Supplier</Label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        list="supplier-bill-options"
                        value={supplierBillSearch}
                        onChange={(e) => handleSupplierBillLookupChange(e.target.value)}
                        placeholder="Search and select supplier"
                        className="h-11 rounded-2xl pl-9"
                      />
                      <datalist id="supplier-bill-options">
                        {supplierBillOptions.map((supplier) => (
                          <option key={supplier.id} value={getSupplierLookupLabel(supplier)} />
                        ))}
                      </datalist>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Start typing supplier name, phone, or email and choose one result.
                    </p>
                    {selectedSupplierForBill && (
                      <div className="grid grid-cols-3 gap-2 rounded-2xl bg-muted/20 p-3 text-xs">
                        <div>
                          <p className="text-muted-foreground">Total</p>
                          <p className="font-semibold">Rs. {formatMoney(selectedSupplierForBill.total_bill)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Paid</p>
                          <p className="font-semibold text-emerald-600">Rs. {formatMoney(selectedSupplierForBill.paid_amount)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Pending</p>
                          <p className="font-semibold text-rose-600">Rs. {formatMoney(selectedSupplierForBill.pending_amount)}</p>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label>Bill Amount</Label>
                    <Input
                      type="number"
                      value={supplierBillAmount}
                      onChange={(e) => setSupplierBillAmount(e.target.value)}
                      className="h-11 rounded-2xl"
                      placeholder="0.00"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Bill Date</Label>
                    <Input
                      type="date"
                      value={supplierBillDate}
                      onChange={(e) => setSupplierBillDate(e.target.value)}
                      className="h-11 rounded-2xl"
                    />
                  </div>

                  <Button className="h-11 w-full rounded-2xl" onClick={addSupplierBill}>
                    Save Supplier Bill
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Dialog
              open={supplierPaymentDialogOpen}
              onOpenChange={(open) => {
                if (!open) {
                  resetSupplierPaymentDialog()
                  return
                }

                setSupplierPaymentDialogOpen(true)
              }}
            >
              <DialogContent className="max-h-[90vh] w-[calc(100vw-1rem)] max-w-[calc(100vw-1rem)] overflow-y-auto rounded-3xl sm:max-w-lg">
                <DialogHeader>
                  <DialogTitle>Pay Supplier Bill</DialogTitle>
                </DialogHeader>
                <div className="space-y-4">
                  <div className="rounded-2xl border bg-muted/20 p-4">
                    <p className="text-sm font-medium">Record supplier payment</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Select a pending bill and the payment will update the bill status and supplier ledger automatically.
                    </p>
                  </div>

                  <div className="space-y-2">
                    <Label>Pending Bill</Label>
                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        list="supplier-payment-options"
                        value={supplierPaymentSearch}
                        onChange={(e) => handleSupplierPaymentLookupChange(e.target.value)}
                        placeholder="Search and select pending bill"
                        className="h-11 rounded-2xl pl-9"
                      />
                      <datalist id="supplier-payment-options">
                        {pendingSupplierInvoiceOptions.map((invoice) => (
                          <option key={invoice.id} value={getSupplierInvoiceLookupLabel(invoice)} />
                        ))}
                      </datalist>
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Only pending supplier bills are shown here.
                    </p>
                    {selectedSupplierPaymentInvoice && (
                      <div className="grid grid-cols-3 gap-2 rounded-2xl bg-muted/20 p-3 text-xs">
                        <div>
                          <p className="text-muted-foreground">Total</p>
                          <p className="font-semibold">Rs. {formatMoney(selectedSupplierPaymentInvoice.total_amount)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Paid</p>
                          <p className="font-semibold text-emerald-600">Rs. {formatMoney(selectedSupplierPaymentInvoice.paid_amount)}</p>
                        </div>
                        <div>
                          <p className="text-muted-foreground">Pending</p>
                          <p className="font-semibold text-rose-600">Rs. {formatMoney(selectedSupplierPaymentInvoice.pending_amount)}</p>
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label>Payment Amount</Label>
                    <Input
                      type="number"
                      value={supplierPaymentAmount}
                      onChange={(e) => setSupplierPaymentAmount(e.target.value)}
                      className="h-11 rounded-2xl"
                      placeholder="0.00"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Payment Mode</Label>
                    <Select value={supplierPaymentMode} onValueChange={setSupplierPaymentMode}>
                      <SelectTrigger className="h-11 rounded-2xl">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="cash">Cash</SelectItem>
                        <SelectItem value="bank">Bank</SelectItem>
                        <SelectItem value="upi">UPI</SelectItem>
                        <SelectItem value="cheque">Cheque</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <Label>Payment Date</Label>
                    <Input
                      type="date"
                      value={supplierPaymentDate}
                      onChange={(e) => setSupplierPaymentDate(e.target.value)}
                      className="h-11 rounded-2xl"
                    />
                  </div>

                  <div className="space-y-2">
                    <Label>Notes</Label>
                    <Input
                      value={supplierPaymentNotes}
                      onChange={(e) => setSupplierPaymentNotes(e.target.value)}
                      className="h-11 rounded-2xl"
                      placeholder="Optional notes"
                    />
                  </div>

                  <Button
                    className="h-11 w-full rounded-2xl"
                    onClick={paySupplierBill}
                    disabled={pendingSupplierInvoices.length === 0}
                  >
                    Record Payment
                  </Button>
                </div>
              </DialogContent>
            </Dialog>

            <Card className="overflow-hidden border-slate-800/70 bg-slate-950/50 shadow-[0_24px_60px_rgba(15,23,42,0.18)]">
              <CardHeader className="border-b border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(96,165,250,0.18),_transparent_28%),radial-gradient(circle_at_top_right,_rgba(129,140,248,0.18),_transparent_30%),linear-gradient(135deg,#020617_0%,#0f172a_54%,#1e293b_100%)] text-white">
                <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-base text-white sm:text-lg">Supplier Directory</CardTitle>
                    <p className="text-sm text-slate-300">
                      Sorted A-Z with quick supplier bill and ledger actions.
                    </p>
                  </div>

                  <div className="flex w-full flex-col gap-2 sm:flex-row lg:w-auto">
                    <div className="relative w-full sm:min-w-[280px]">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                      <Input
                        value={supplierSearch}
                        onChange={(e) => setSupplierSearch(e.target.value)}
                        placeholder="Search supplier by name, phone, email, or address"
                        className="h-11 rounded-2xl border-white/10 bg-white/10 pl-9 text-white placeholder:text-slate-300"
                      />
                    </div>
                    <Button
                      variant="outline"
                      className="h-11 rounded-2xl border-white/15 bg-white/10 text-white hover:bg-white/15 hover:text-white"
                      onClick={() => openSupplierBillDialogFor()}
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Add Bill
                    </Button>
                    <Button
                      className="h-11 rounded-2xl bg-indigo-500 text-white shadow-[0_14px_30px_rgba(99,102,241,0.32)] hover:bg-indigo-400"
                      onClick={() => openSupplierPaymentDialogFor()}
                      disabled={pendingSupplierInvoices.length === 0}
                    >
                      <Wallet className="mr-2 h-4 w-4" />
                      Pay Bill
                    </Button>
                  </div>
                </div>
              </CardHeader>

              <CardContent className="space-y-4 p-4">
                <div className="grid gap-4 md:hidden">
                  {sortedFilteredSuppliers.length === 0 ? (
                    <div className="rounded-3xl border border-dashed p-10 text-center text-sm text-muted-foreground">
                      No suppliers found
                    </div>
                  ) : (
                    sortedFilteredSuppliers.map((supplier) => (
                      <Card key={supplier.id} className="rounded-[28px] border border-white/10 bg-slate-950/70 shadow-[0_18px_40px_rgba(15,23,42,0.22)]">
                        <CardContent className="space-y-3 p-3.5 text-white">
                          <div className="flex items-start justify-between gap-3">
                            <div className="flex items-start gap-3">
                              <div className="flex h-10 w-10 items-center justify-center rounded-2xl bg-white/10 text-white">
                                <Building2 className="h-4 w-4" />
                              </div>
                              <div>
                                <p className="text-sm font-semibold">{supplier.name}</p>
                                {supplier.phone ? (
                                  <p className="text-xs text-slate-300">{supplier.phone}</p>
                                ) : null}
                              </div>
                            </div>
                            <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ${Number(supplier.pending_amount || 0) > 0 ? "bg-rose-500/20 text-rose-200" : "bg-emerald-500/20 text-emerald-200"}`}>
                              {Number(supplier.pending_amount || 0) > 0 ? "Pending" : "Settled"}
                            </span>
                          </div>

                          {(supplier.email || supplier.address) && (
                            <div className="space-y-1 text-xs text-slate-400">
                              {supplier.email ? <p className="truncate">{supplier.email}</p> : null}
                              {supplier.address ? <p className="break-words leading-4">{supplier.address}</p> : null}
                            </div>
                          )}

                          <div className="grid grid-cols-2 gap-2 rounded-2xl bg-white/5 p-2.5 text-[11px]">
                            <div>
                              <p className="text-slate-400">Total Bill</p>
                              <p className="font-semibold">Rs. {formatMoney(supplier.total_bill)}</p>
                            </div>
                            <div>
                              <p className="text-slate-400">Balance</p>
                              <p className={`font-semibold ${Number(supplier.current_balance || 0) > 0 ? "text-rose-300" : "text-emerald-300"}`}>
                                Rs. {formatMoney(supplier.current_balance)}
                              </p>
                            </div>
                            <div>
                              <p className="text-slate-400">Paid</p>
                              <p className="font-semibold text-emerald-300">Rs. {formatMoney(supplier.paid_amount)}</p>
                            </div>
                            <div>
                              <p className="text-slate-400">Pending</p>
                              <p className="font-semibold text-rose-300">Rs. {formatMoney(supplier.pending_amount)}</p>
                            </div>
                          </div>

                          <div className="grid grid-cols-2 gap-2">
                            <Button
                              variant="outline"
                              className="h-10 rounded-2xl border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white"
                              onClick={() => openSupplierBillDialogFor(supplier)}
                            >
                              Add Bill
                            </Button>
                            <Button
                              className="h-10 rounded-2xl bg-indigo-500 text-white hover:bg-indigo-400"
                              onClick={() => openSupplierDetails(supplier)}
                            >
                              Ledger Entry
                            </Button>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </div>

                <div className="hidden overflow-x-auto md:block">
                  <Table>
                    <TableHeader>
                      <TableRow className="bg-muted/50">
                        <TableHead>#</TableHead>
                        <TableHead>Supplier</TableHead>
                        <TableHead>Contact</TableHead>
                        <TableHead className="text-right">Total Bill</TableHead>
                        <TableHead className="text-right">Paid</TableHead>
                        <TableHead className="text-right">Pending</TableHead>
                        <TableHead className="text-right">Balance</TableHead>
                        <TableHead>Address</TableHead>
                        <TableHead className="text-right">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedFilteredSuppliers.length === 0 ? (
                        <TableRow>
                          <TableCell colSpan="9" className="py-8 text-center text-muted-foreground">
                            No suppliers found
                          </TableCell>
                        </TableRow>
                      ) : (
                        sortedFilteredSuppliers.map((supplier, index) => (
                          <TableRow key={supplier.id}>
                            <TableCell>{index + 1}</TableCell>
                            <TableCell>
                              <div>
                                <p className="font-medium">{supplier.name}</p>
                                <p className="text-xs text-muted-foreground">{supplier.email || "-"}</p>
                              </div>
                            </TableCell>
                            <TableCell>{supplier.phone || "-"}</TableCell>
                            <TableCell className="text-right font-semibold">Rs. {formatMoney(supplier.total_bill)}</TableCell>
                            <TableCell className="text-right font-semibold text-emerald-600">Rs. {formatMoney(supplier.paid_amount)}</TableCell>
                            <TableCell className="text-right font-semibold text-rose-600">Rs. {formatMoney(supplier.pending_amount)}</TableCell>
                            <TableCell className={`text-right font-semibold ${Number(supplier.current_balance || 0) > 0 ? "text-rose-600" : "text-emerald-600"}`}>
                              Rs. {formatMoney(supplier.current_balance)}
                            </TableCell>
                            <TableCell className="max-w-xs truncate">{supplier.address || "-"}</TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-2">
                                <Button
                                  size="sm"
                                  variant="outline"
                                  className="rounded-xl"
                                  onClick={() => openSupplierBillDialogFor(supplier)}
                                >
                                  Add Bill
                                </Button>
                                <Button
                                  size="sm"
                                  className="rounded-xl"
                                  onClick={() => openSupplierDetails(supplier)}
                                >
                                  Ledger Entry
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))
                      )}
                    </TableBody>
                  </Table>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        <Dialog
          open={!!viewSupplier}
          onOpenChange={(open) => {
            if (!open) {
              setViewSupplier(null)
              setSupplierLedger(null)
              setSupplierLedgerSearch("")
              setSupplierDetailTab("ledger")
            }
          }}
        >
          <DialogContent className="max-h-[90vh] max-w-[95vw] overflow-y-auto rounded-3xl sm:max-w-6xl">
            <DialogHeader>
              <DialogTitle>Supplier Ledger</DialogTitle>
            </DialogHeader>
            {viewSupplierDetails && (
              <div className="space-y-5 text-sm">
                <div className="overflow-hidden rounded-3xl bg-[linear-gradient(135deg,#0f172a_0%,#1e293b_45%,#334155_100%)] text-white shadow-sm">
                  <div className="flex flex-col gap-6 p-5 lg:flex-row lg:items-start lg:justify-between">
                    <div className="space-y-3">
                      <div>
                        <p className="text-xs uppercase tracking-[0.28em] text-slate-300">Supplier Ledger</p>
                        <p className="mt-2 text-2xl font-semibold">{viewSupplierDetails.name}</p>
                      </div>
                      <div className="space-y-1 text-sm text-slate-200">
                        <p>Phone: {viewSupplierDetails.phone || "-"}</p>
                        <p>Email: {viewSupplierDetails.email || "-"}</p>
                        <p>Address: {viewSupplierDetails.address || "-"}</p>
                      </div>
                      <div className="flex flex-wrap gap-2 pt-1">
                        <Button
                          variant="secondary"
                          className="rounded-2xl"
                          onClick={() => openSupplierBillDialogFor(viewSupplierDetails)}
                        >
                          <Plus className="mr-2 h-4 w-4" />
                          Add Bill
                        </Button>
                        <Button
                          variant="outline"
                          className="rounded-2xl border-white/20 bg-white/10 text-white hover:bg-white/15 hover:text-white"
                          onClick={() => openSupplierPaymentDialogFor(viewSupplierDetails)}
                          disabled={viewSupplierOutstanding <= 0}
                        >
                          <Wallet className="mr-2 h-4 w-4" />
                          Pay Bill
                        </Button>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3 lg:min-w-[26rem]">
                      <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Total Bill</p>
                        <p className="mt-2 text-xl font-semibold">Rs. {formatMoney(viewSupplierDetails.total_bill)}</p>
                      </div>
                      <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Paid</p>
                        <p className="mt-2 text-xl font-semibold text-emerald-300">Rs. {formatMoney(viewSupplierDetails.paid_amount)}</p>
                      </div>
                      <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Pending</p>
                        <p className="mt-2 text-xl font-semibold text-amber-200">Rs. {formatMoney(viewSupplierDetails.pending_amount)}</p>
                      </div>
                      <div className="rounded-2xl bg-white/10 p-4 backdrop-blur">
                        <p className="text-xs uppercase tracking-[0.18em] text-slate-300">Ledger Balance</p>
                        <p className={`mt-2 text-xl font-semibold ${viewSupplierComputedBalance > 0 ? "text-rose-200" : viewSupplierComputedBalance < 0 ? "text-emerald-200" : "text-white"}`}>
                          Rs. {formatMoney(viewSupplierComputedBalance)}
                        </p>
                      </div>
                    </div>
                  </div>
                </div>

                {supplierLedgerLoading ? (
                  <div className="rounded-2xl border border-dashed p-8 text-center text-muted-foreground">
                    Loading supplier ledger...
                  </div>
                ) : (
                  <>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <Button
                        variant={supplierDetailTab === "ledger" ? "default" : "outline"}
                        className={`h-11 rounded-2xl ${
                          supplierDetailTab === "ledger"
                            ? "bg-slate-900 text-white hover:bg-slate-800"
                            : "border-white/10 bg-slate-950/50 text-slate-300 hover:bg-white/5 hover:text-white"
                        }`}
                        onClick={() => setSupplierDetailTab("ledger")}
                      >
                        Ledger Entries
                      </Button>
                      <Button
                        variant={supplierDetailTab === "bills" ? "default" : "outline"}
                        className={`h-11 rounded-2xl ${
                          supplierDetailTab === "bills"
                            ? "bg-slate-900 text-white hover:bg-slate-800"
                            : "border-white/10 bg-slate-950/50 text-slate-300 hover:bg-white/5 hover:text-white"
                        }`}
                        onClick={() => setSupplierDetailTab("bills")}
                      >
                        Supplier Bills
                      </Button>
                    </div>

                    <div className="relative">
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                      <Input
                        placeholder={
                          supplierDetailTab === "ledger"
                            ? "Search ledger date, type, mode, amount, or note..."
                            : "Search supplier bill number, date, status, or amount..."
                        }
                        value={supplierLedgerSearch}
                        onChange={(e) => setSupplierLedgerSearch(e.target.value)}
                        className="h-11 rounded-2xl pl-9"
                      />
                    </div>

                    {supplierDetailTab === "ledger" ? (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between gap-3">
                          <h3 className="text-base font-semibold">Ledger Entries</h3>
                          <p className="text-xs text-muted-foreground">
                            {filteredSupplierLedgerEntries.length} of {supplierLedgerEntries.length} row(s)
                          </p>
                        </div>
                        <div className="max-h-[24rem] overflow-y-auto rounded-2xl border p-3">
                          {filteredSupplierLedgerEntries.length === 0 ? (
                            <p className="text-sm text-muted-foreground">No supplier ledger entries found</p>
                          ) : (
                            <div className="overflow-x-auto rounded-xl border">
                              <Table>
                                <TableHeader>
                                  <TableRow className="bg-muted/30">
                                    <TableHead>Date</TableHead>
                                    <TableHead>Time</TableHead>
                                    <TableHead>Type</TableHead>
                                    <TableHead className="text-right">Debit</TableHead>
                                    <TableHead className="text-right">Credit</TableHead>
                                    <TableHead className="text-right">Balance</TableHead>
                                    <TableHead>Mode</TableHead>
                                    <TableHead>Details</TableHead>
                                  </TableRow>
                                </TableHeader>
                                <TableBody>
                                  {filteredSupplierLedgerEntries.map((entry, index) => (
                                    <TableRow key={entry.id || `${entry.date || "entry"}-${index}`}>
                                      <TableCell>{formatDateValue(entry.date || entry.created_at)}</TableCell>
                                      <TableCell>{formatTimeValue(entry.created_at)}</TableCell>
                                      <TableCell>
                                        <div className="space-y-1">
                                          <p className="font-medium">{getSupplierLedgerTypeLabel(entry.type)}</p>
                                          {entry.bill_number && (
                                            <p className="text-xs text-muted-foreground">{entry.bill_number}</p>
                                          )}
                                        </div>
                                      </TableCell>
                                      <TableCell className="text-right font-medium text-rose-600">
                                        {Number(entry.debit || 0) > 0 ? `Rs. ${formatMoney(entry.debit)}` : "-"}
                                      </TableCell>
                                      <TableCell className="text-right font-medium text-emerald-600">
                                        {Number(entry.credit || 0) > 0 ? `Rs. ${formatMoney(entry.credit)}` : "-"}
                                      </TableCell>
                                      <TableCell className={`text-right font-semibold ${Number(entry.balance || 0) > 0 ? "text-rose-600" : Number(entry.balance || 0) < 0 ? "text-emerald-600" : "text-foreground"}`}>
                                        Rs. {formatMoney(entry.balance)}
                                      </TableCell>
                                      <TableCell className="uppercase text-muted-foreground">{entry.mode || "-"}</TableCell>
                                      <TableCell className="max-w-sm">
                                        <div className="space-y-1">
                                          <p>{entry.description || "-"}</p>
                                          {entry.status && (
                                            <span className={`inline-flex rounded-full px-2 py-0.5 text-[11px] font-semibold ${getInvoiceStatusClass(entry.status)}`}>
                                              {String(entry.status).toUpperCase()}
                                            </span>
                                          )}
                                        </div>
                                      </TableCell>
                                    </TableRow>
                                  ))}
                                </TableBody>
                              </Table>
                            </div>
                          )}
                        </div>
                      </div>
                    ) : (
                      <div className="space-y-3">
                        <div className="flex items-center justify-between gap-3">
                          <h3 className="text-base font-semibold">Supplier Bills</h3>
                          <p className="text-xs text-muted-foreground">
                            {filteredSupplierInvoices.length} of {supplierLedgerInvoices.length} bill(s)
                          </p>
                        </div>
                        {filteredSupplierInvoices.length === 0 ? (
                          <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 p-8 text-center text-slate-400">
                            No supplier bills found
                          </div>
                        ) : (
                          <div className="grid gap-4 lg:grid-cols-2">
                            {filteredSupplierInvoices.map((invoice) => (
                              <div
                                key={invoice.id}
                                className="rounded-3xl border border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.14),_transparent_28%),linear-gradient(135deg,#020617_0%,#0f172a_54%,#111827_100%)] p-5 text-white shadow-[0_18px_48px_rgba(15,23,42,0.24)]"
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div>
                                    <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Supplier Bill</p>
                                    <p className="mt-2 text-lg font-semibold">{invoice.bill_number || invoice.id}</p>
                                    <p className="mt-1 text-sm text-slate-300">
                                      Issue Date: {formatDateValue(invoice.invoice_date)}
                                    </p>
                                  </div>
                                  <span className={`rounded-full px-3 py-1 text-xs font-semibold ${getInvoiceStatusClass(invoice.status)}`}>
                                    {String(invoice.status || "pending").toUpperCase()}
                                  </span>
                                </div>

                                <div className="mt-5 rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                                  <div className="flex items-start justify-between gap-3 border-b border-white/10 pb-3">
                                    <div>
                                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Supplier</p>
                                      <p className="mt-1 font-semibold">{invoice.supplier_name || viewSupplierDetails.name}</p>
                                    </div>
                                    <div className="text-right">
                                      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Created</p>
                                      <p className="mt-1 text-sm text-slate-200">{formatDateValue(invoice.created_at || invoice.invoice_date)}</p>
                                    </div>
                                  </div>

                                  <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-4">
                                    <div className="rounded-2xl bg-white/5 p-3">
                                      <p className="text-xs uppercase tracking-[0.14em] text-slate-400">Type</p>
                                      <p className="mt-2 text-sm font-semibold">{getSupplierLedgerTypeLabel("supplier_bill")}</p>
                                    </div>
                                    <div className="rounded-2xl bg-white/5 p-3">
                                      <p className="text-xs uppercase tracking-[0.14em] text-slate-400">Total</p>
                                      <p className="mt-2 text-lg font-semibold">Rs. {formatMoney(invoice.total_amount)}</p>
                                    </div>
                                    <div className="rounded-2xl bg-emerald-500/10 p-3">
                                      <p className="text-xs uppercase tracking-[0.14em] text-emerald-300">Paid</p>
                                      <p className="mt-2 text-lg font-semibold text-emerald-300">Rs. {formatMoney(invoice.paid_amount)}</p>
                                    </div>
                                    <div className="rounded-2xl bg-rose-500/10 p-3">
                                      <p className="text-xs uppercase tracking-[0.14em] text-rose-300">Pending</p>
                                      <p className="mt-2 text-lg font-semibold text-rose-300">Rs. {formatMoney(invoice.pending_amount)}</p>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}


