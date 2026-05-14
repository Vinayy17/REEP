"use client"

import { useEffect, useMemo, useState } from "react"
import axios from "axios"
import { useLocation, useNavigate } from "react-router-dom"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
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
import EntityCard from "@/components/people/EntityCard"
import StatsCardGrid from "@/components/people/StatsCardGrid"
import {
  Building2,
  Edit,
  FileText,
  Plus,
  Search,
  SlidersHorizontal,
  Trash2,
  User,
  Wallet,
} from "lucide-react"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const emptyCustomer = {
  name: "",
  email: "",
  phone: "",
  address: "",
}

const emptySupplier = {
  name: "",
  email: "",
  phone: "",
  address: "",
  opening_balance: 0,
}

const createEmptyAdvanceForm = () => ({
  amount: "",
  payment_mode: "cash",
  reference: "",
  payment_date: new Date().toISOString().slice(0, 10),
})

function formatMoney(value) {
  return Number(value || 0).toLocaleString("en-IN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
}

function formatCurrency(value) {
  return `\u20B9\u00A0${formatMoney(value)}`
}

function formatDateValue(value) {
  if (!value) return "-"

  try {
    return new Date(value).toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
  } catch {
    return value
  }
}

function formatTimeValue(value) {
  if (!value) return "-"

  try {
    return new Date(value).toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
    })
  } catch {
    return "-"
  }
}

function formatDateTimeValue(value) {
  if (!value) return "-"

  try {
    const date = new Date(value)
    return {
      date: date.toLocaleDateString("en-IN", {
        day: "2-digit",
        month: "short",
        year: "numeric",
      }),
      time: date.toLocaleTimeString("en-IN", {
        hour: "2-digit",
        minute: "2-digit",
      }),
    }
  } catch {
    return { date: "-", time: "-" }
  }
}

function matchesSearchText(parts, query) {
  const normalized = String(query || "").trim().toLowerCase()
  if (!normalized) return true

  return parts
    .filter(Boolean)
    .some((part) => String(part).toLowerCase().includes(normalized))
}

function getCustomerBalanceText(customer) {
  const pending = Number(customer?.pending_balance || customer?.pending_invoice_amount || 0)
  const advance = Number(customer?.advance_balance || 0)

  if (pending > 0) return `Pending ${formatCurrency(pending)}`
  if (advance > 0) return `Advance ${formatCurrency(advance)}`
  return "No pending or advance"
}

function getCustomerBalanceSummary(customer) {
  const pending = Number(customer?.pending_balance || customer?.pending_invoice_amount || 0)
  const advance = Number(customer?.advance_balance || 0)

  if (pending > 0) return formatCurrency(pending)
  if (advance > 0) return formatCurrency(advance)
  return "Clear"
}

function getCustomerStatus(customer) {
  const pending = Number(customer?.pending_balance || customer?.pending_invoice_amount || 0)
  const advance = Number(customer?.advance_balance || 0)

  if (pending > 0) {
    return { label: "Pending", tone: "warning" }
  }

  if (advance > 0) {
    return { label: "Advance", tone: "info" }
  }

  return { label: "Settled", tone: "success" }
}

function getSupplierStatus(supplier) {
  return Number(supplier?.pending_amount || 0) > 0
    ? { label: "Pending", tone: "warning" }
    : { label: "Settled", tone: "success" }
}

function getSupplierLedgerTypeLabel(type) {
  const tone = String(type || "").toLowerCase()

  if (tone === "opening_balance") return "Opening"
  if (tone === "supplier_bill") return "Bill Added"
  if (tone === "supplier_payment") return "Payment Made"
  if (tone === "payment_out") return "Payment Out"

  return String(type || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function getCustomerLedgerTypeLabel(type) {
  const normalized = String(type || "").toLowerCase()

  if (normalized === "payment") return "Payment Received"
  if (normalized === "payment_in") return "Payment Received"
  if (normalized === "sale_due") return "Sale Due"
  if (normalized === "advance_in") return "Advance Received"
  if (normalized === "advance_used") return "Advance Used"
  if (normalized === "sale_payment") return "Sale Payment"

  return String(type || "-")
    .replaceAll("_", " ")
    .replace(/\b\w/g, (char) => char.toUpperCase())
}

function getInvoiceStatusClass(status) {
  const normalized = String(status || "pending").toLowerCase()

  if (normalized === "paid") {
    return "border-emerald-400/20 bg-emerald-500/15 text-emerald-200"
  }

  if (normalized === "partial") {
    return "border-amber-400/20 bg-amber-500/15 text-amber-200"
  }

  return "border-rose-400/20 bg-rose-500/15 text-rose-200"
}

function getSupplierLookupLabel(supplier) {
  return [supplier?.name, supplier?.phone, supplier?.email].filter(Boolean).join(" - ")
}

function getSupplierInvoiceLookupLabel(invoice) {
  return [
    invoice?.bill_number || invoice?.id,
    invoice?.supplier_name,
    `Pending ${formatCurrency(invoice?.pending_amount)}`,
  ]
    .filter(Boolean)
    .join(" - ")
}

function sortByName(items) {
  return [...items].sort((leftItem, rightItem) =>
    String(leftItem?.name || "").localeCompare(String(rightItem?.name || ""), "en", {
      sensitivity: "base",
    })
  )
}

const floatingCardClass =
  "border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.032),rgba(255,255,255,0.014))] backdrop-blur-xl shadow-[0_18px_40px_rgba(2,6,23,0.1)]"

const floatingPanelClass =
  "rounded-[22px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.028),rgba(255,255,255,0.012))] backdrop-blur-xl shadow-[0_18px_40px_rgba(2,6,23,0.1)]"

const desktopTableShellClass =
  "overflow-hidden rounded-[22px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.008))] backdrop-blur-xl shadow-[0_18px_40px_rgba(2,6,23,0.1)]"

export default function Customers() {
  const location = useLocation()
  const navigate = useNavigate()

  const searchParams = new URLSearchParams(location.search)
  const activeTab = searchParams.get("tab") === "suppliers" ? "suppliers" : "customers"

  const [customers, setCustomers] = useState([])
  const [suppliers, setSuppliers] = useState([])
  const [customerSearch, setCustomerSearch] = useState("")
  const [supplierSearch, setSupplierSearch] = useState("")
  const [customerSort, setCustomerSort] = useState("az")
  const [customerBalanceFilter, setCustomerBalanceFilter] = useState("all")

  const [openCustomer, setOpenCustomer] = useState(false)
  const [openSupplier, setOpenSupplier] = useState(false)
  const [editingCustomer, setEditingCustomer] = useState(null)
  const [customerForm, setCustomerForm] = useState(emptyCustomer)
  const [supplierForm, setSupplierForm] = useState(emptySupplier)

  const [viewCustomer, setViewCustomer] = useState(null)
  const [customerLedger, setCustomerLedger] = useState(null)
  const [customerLedgerLoading, setCustomerLedgerLoading] = useState(false)
  const [customerDetailTab, setCustomerDetailTab] = useState("ledger")
  const [customerDetailSearch, setCustomerDetailSearch] = useState("")

  const [advanceDialogOpen, setAdvanceDialogOpen] = useState(false)
  const [advanceCustomer, setAdvanceCustomer] = useState(null)
  const [advanceForm, setAdvanceForm] = useState(createEmptyAdvanceForm())
  const [advanceSubmitting, setAdvanceSubmitting] = useState(false)

  const [viewSupplierDetails, setViewSupplierDetails] = useState(null)
  const [supplierLedger, setSupplierLedger] = useState(null)
  const [supplierLedgerLoading, setSupplierLedgerLoading] = useState(false)
  const [supplierLedgerSearch, setSupplierLedgerSearch] = useState("")
  const [supplierDetailTab, setSupplierDetailTab] = useState("ledger")
  const [supplierSnapshotCache, setSupplierSnapshotCache] = useState({})

  const [supplierBillDialogOpen, setSupplierBillDialogOpen] = useState(false)
  const [supplierBillSupplierId, setSupplierBillSupplierId] = useState("")
  const [supplierBillSearch, setSupplierBillSearch] = useState("")
  const [supplierBillAmount, setSupplierBillAmount] = useState("")
  const [supplierBillDate, setSupplierBillDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [supplierBillDropdownOpen, setSupplierBillDropdownOpen] = useState(false)

  const [supplierPaymentDialogOpen, setSupplierPaymentDialogOpen] = useState(false)
  const [supplierPaymentSupplierId, setSupplierPaymentSupplierId] = useState("")
  const [supplierPaymentSupplierSearch, setSupplierPaymentSupplierSearch] = useState("")
  const [supplierPaymentSupplierDropdownOpen, setSupplierPaymentSupplierDropdownOpen] = useState(false)
  const [supplierPaymentInvoiceId, setSupplierPaymentInvoiceId] = useState("")
  const [supplierPaymentInvoiceSearch, setSupplierPaymentInvoiceSearch] = useState("")
  const [supplierPaymentInvoiceDropdownOpen, setSupplierPaymentInvoiceDropdownOpen] = useState(false)
  const [supplierPaymentInvoices, setSupplierPaymentInvoices] = useState([])
  const [supplierPaymentAmount, setSupplierPaymentAmount] = useState("")
  const [supplierPaymentMode, setSupplierPaymentMode] = useState("cash")
  const [supplierPaymentNotes, setSupplierPaymentNotes] = useState("")
  const [supplierPaymentDate, setSupplierPaymentDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [supplierPaymentLoading, setSupplierPaymentLoading] = useState(false)

  const tokenHeaders = () => ({
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`,
    },
  })

  const clearActionParam = () => {
    const params = new URLSearchParams(location.search)
    params.delete("action")

    navigate(
      {
        pathname: "/customers",
        search: params.toString() ? `?${params.toString()}` : "",
      },
      { replace: true }
    )
  }

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

  useEffect(() => {
    fetchCustomers()
    fetchSuppliers()
  }, [])

  useEffect(() => {
    if (searchParams.get("action") !== "add") return

    if (activeTab === "suppliers") {
      setOpenSupplier(true)
    } else {
      setOpenCustomer(true)
    }

    clearActionParam()
  }, [activeTab, location.search])

  const fetchCustomerDetails = async (customerId) => {
    const res = await axios.get(`${API}/accounts/customer-ledger/${customerId}`, tokenHeaders())
    return res.data
  }

  const fetchSupplierSnapshot = async (supplierId, forceRefresh = false) => {
    if (!supplierId) return null

    if (!forceRefresh && supplierSnapshotCache[supplierId]) {
      return supplierSnapshotCache[supplierId]
    }

    const res = await axios.get(`${API}/accounts/supplier-ledger/${supplierId}`, tokenHeaders())
    setSupplierSnapshotCache((current) => ({
      ...current,
      [supplierId]: res.data,
    }))
    return res.data
  }

  const refreshCustomerSection = async (customerId = null) => {
    const freshCustomers = await fetchCustomers()

    if (customerId && viewCustomer?.id === customerId) {
      try {
        const ledgerData = await fetchCustomerDetails(customerId)
        const matchingCustomer = freshCustomers.find((customer) => customer.id === customerId)
        setCustomerLedger(ledgerData)
        setViewCustomer({
          ...matchingCustomer,
          ...ledgerData.customer,
        })
      } catch {
        // keep current modal state if refresh fails
      }
    }

    return freshCustomers
  }

  const hydrateSupplierPaymentInvoices = (invoices, keepInvoiceId = null) => {
    const pendingInvoices = (invoices || []).filter((invoice) => Number(invoice.pending_amount || 0) > 0)
    setSupplierPaymentInvoices(pendingInvoices)

    const nextSelected =
      pendingInvoices.find((invoice) => invoice.id === keepInvoiceId) || pendingInvoices[0] || null

    setSupplierPaymentInvoiceId(nextSelected?.id || "")
    setSupplierPaymentInvoiceSearch(nextSelected ? getSupplierInvoiceLookupLabel(nextSelected) : "")
    setSupplierPaymentAmount(nextSelected ? String(Number(nextSelected.pending_amount || 0)) : "")
    setSupplierPaymentInvoiceDropdownOpen(false)
  }

  const refreshSupplierSection = async (supplierId = null) => {
    const freshSuppliers = await fetchSuppliers()

    if (supplierId) {
      try {
        const snapshot = await fetchSupplierSnapshot(supplierId, true)

        if (viewSupplierDetails?.id === supplierId) {
          setSupplierLedger(snapshot)
          setViewSupplierDetails({
            ...(freshSuppliers.find((supplier) => supplier.id === supplierId) || viewSupplierDetails),
            ...snapshot.supplier,
          })
        }

        if (supplierPaymentSupplierId === supplierId) {
          hydrateSupplierPaymentInvoices(snapshot.invoices, supplierPaymentInvoiceId)
        }
      } catch {
        // leave existing UI data as-is if refresh fails
      }
    }

    return freshSuppliers
  }

  const resetCustomerForm = () => {
    setEditingCustomer(null)
    setCustomerForm(emptyCustomer)
  }

  const resetSupplierForm = () => {
    setSupplierForm(emptySupplier)
  }

  const submitCustomer = async (event) => {
    event.preventDefault()

    try {
      if (editingCustomer) {
        await axios.put(`${API}/customers/${editingCustomer.id}`, customerForm, tokenHeaders())
        toast.success("Customer updated successfully")
      } else {
        await axios.post(`${API}/customers`, customerForm, tokenHeaders())
        toast.success("Customer added successfully")
      }

      await refreshCustomerSection(editingCustomer?.id || null)
      setOpenCustomer(false)
      resetCustomerForm()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to save customer")
    }
  }

  const submitSupplier = async (event) => {
    event.preventDefault()

    try {
      await axios.post(`${API}/suppliers`, supplierForm, tokenHeaders())
      toast.success("Supplier added successfully")
      await refreshSupplierSection()
      setOpenSupplier(false)
      resetSupplierForm()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to save supplier")
    }
  }

  const deleteCustomer = async (customer) => {
    if (!window.confirm(`Delete ${customer.name}?`)) return

    try {
      await axios.delete(`${API}/customers/${customer.id}`, tokenHeaders())
      toast.success("Customer deleted successfully")
      await refreshCustomerSection()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to delete customer")
    }
  }

  const handleEditCustomer = (customer) => {
    setEditingCustomer(customer)
    setCustomerForm({
      name: customer.name || "",
      email: customer.email || "",
      phone: customer.phone || "",
      address: customer.address || "",
    })
    setOpenCustomer(true)
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
          current_balance: Number(customer.current_balance || 0),
        },
      },
    })
  }

  const openCustomerDetails = async (customer) => {
    setViewCustomer(customer)
    setCustomerLedger(null)
    setCustomerLedgerLoading(true)
    setCustomerDetailTab("ledger")
    setCustomerDetailSearch("")

    try {
      const details = await fetchCustomerDetails(customer.id)
      const matchingCustomer = customers.find((row) => row.id === customer.id)

      setCustomerLedger(details)
      setViewCustomer({
        ...matchingCustomer,
        ...details.customer,
      })
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to load customer ledger")
    } finally {
      setCustomerLedgerLoading(false)
    }
  }

  const openAdvanceDialogFor = (customer) => {
    if (!customer?.can_add_advance) {
      toast.error("Advance cannot be added while pending balance exists for this customer")
      return
    }

    setAdvanceCustomer(customer)
    setAdvanceForm(createEmptyAdvanceForm())
    setAdvanceDialogOpen(true)
  }

  const submitAdvance = async (event) => {
    event.preventDefault()

    if (!advanceCustomer) return

    const amount = Number(advanceForm.amount || 0)
    if (amount <= 0) {
      toast.error("Advance amount must be greater than zero")
      return
    }

    setAdvanceSubmitting(true)

    try {
      await axios.post(
        `${API}/customers/${advanceCustomer.id}/advance`,
        {
          amount,
          payment_mode: advanceForm.payment_mode,
          reference: advanceForm.reference || null,
          payment_date: advanceForm.payment_date || null,
        },
        tokenHeaders()
      )

      toast.success("Advance added successfully")
      setAdvanceDialogOpen(false)
      await refreshCustomerSection(advanceCustomer.id)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add advance")
    } finally {
      setAdvanceSubmitting(false)
    }
  }

  const openSupplierDetails = async (supplier) => {
    setViewSupplierDetails(supplier)
    setSupplierLedger(null)
    setSupplierLedgerLoading(true)
    setSupplierLedgerSearch("")
    setSupplierDetailTab("ledger")

    try {
      const snapshot = await fetchSupplierSnapshot(supplier.id, true)
      const matchingSupplier = suppliers.find((row) => row.id === supplier.id)

      setSupplierLedger(snapshot)
      setViewSupplierDetails({
        ...matchingSupplier,
        ...snapshot.supplier,
      })
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to load supplier details")
    } finally {
      setSupplierLedgerLoading(false)
    }
  }

  const resetSupplierBillDialog = () => {
    setSupplierBillDialogOpen(false)
    setSupplierBillSupplierId("")
    setSupplierBillSearch("")
    setSupplierBillAmount("")
    setSupplierBillDate(new Date().toISOString().slice(0, 10))
    setSupplierBillDropdownOpen(false)
  }

  const openSupplierBillDialogFor = (supplier = null) => {
    setSupplierBillSupplierId(supplier?.id || "")
    setSupplierBillSearch(supplier ? getSupplierLookupLabel(supplier) : "")
    setSupplierBillAmount("")
    setSupplierBillDate(new Date().toISOString().slice(0, 10))
    setSupplierBillDropdownOpen(true)
    setSupplierBillDialogOpen(true)
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

  const resetSupplierPaymentDialog = () => {
    setSupplierPaymentDialogOpen(false)
    setSupplierPaymentSupplierId("")
    setSupplierPaymentSupplierSearch("")
    setSupplierPaymentSupplierDropdownOpen(false)
    setSupplierPaymentInvoiceId("")
    setSupplierPaymentInvoiceSearch("")
    setSupplierPaymentInvoiceDropdownOpen(false)
    setSupplierPaymentInvoices([])
    setSupplierPaymentAmount("")
    setSupplierPaymentMode("cash")
    setSupplierPaymentNotes("")
    setSupplierPaymentDate(new Date().toISOString().slice(0, 10))
    setSupplierPaymentLoading(false)
  }

  const selectSupplierForPayment = async (supplier) => {
    if (!supplier) return

    setSupplierPaymentSupplierId(supplier.id)
    setSupplierPaymentSupplierSearch(getSupplierLookupLabel(supplier))
    setSupplierPaymentSupplierDropdownOpen(false)
    setSupplierPaymentLoading(true)

    try {
      const snapshot = await fetchSupplierSnapshot(supplier.id, true)
      hydrateSupplierPaymentInvoices(snapshot?.invoices || [])

      if (!snapshot?.invoices?.some((invoice) => Number(invoice.pending_amount || 0) > 0)) {
        toast.error("No pending balance available for this supplier")
      }
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to load supplier bills")
    } finally {
      setSupplierPaymentLoading(false)
    }
  }

  const openSupplierPaymentDialogFor = async (supplier = null) => {
    resetSupplierPaymentDialog()
    setSupplierPaymentDialogOpen(true)

    if (supplier) {
      await selectSupplierForPayment(supplier)
    } else {
      setSupplierPaymentSupplierDropdownOpen(true)
    }
  }

  const paySupplierBill = async () => {
    if (!supplierPaymentInvoiceId || !supplierPaymentAmount) {
      toast.error("Select a pending bill and enter amount")
      return
    }

    const amount = Number(supplierPaymentAmount || 0)
    if (amount <= 0) {
      toast.error("Payment amount must be greater than zero")
      return
    }

    const targetInvoice = supplierPaymentInvoices.find((invoice) => invoice.id === supplierPaymentInvoiceId)

    try {
      await axios.post(
        `${API}/supplier-invoices/${supplierPaymentInvoiceId}/pay`,
        {
          amount,
          payment_mode: supplierPaymentMode,
          payment_date: supplierPaymentDate || null,
          notes: supplierPaymentNotes || null,
        },
        tokenHeaders()
      )

      toast.success("Supplier bill paid successfully")
      resetSupplierPaymentDialog()
      await refreshSupplierSection(targetInvoice?.supplier_id || supplierPaymentSupplierId || null)
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to pay supplier bill")
    }
  }

  const filteredCustomers = useMemo(() => {
    const query = customerSearch.trim().toLowerCase()

    return customers.filter((customer) => {
      const matchesSearch = !query
        ? true
        : [customer.name, customer.email, customer.phone, customer.address]
            .filter(Boolean)
            .some((value) => String(value).toLowerCase().includes(query))

      if (!matchesSearch) return false

      if (customerBalanceFilter === "pending") {
        return Number(customer.pending_balance || customer.pending_invoice_amount || 0) > 0
      }

      if (customerBalanceFilter === "advance") {
        return Number(customer.advance_balance || 0) > 0
      }

      return true
    })
  }, [customerBalanceFilter, customerSearch, customers])

  const sortedCustomers = useMemo(() => {
    const rows = sortByName(filteredCustomers)
    return customerSort === "za" ? [...rows].reverse() : rows
  }, [customerSort, filteredCustomers])

  const filteredSuppliers = useMemo(() => {
    const query = supplierSearch.trim().toLowerCase()

    return suppliers.filter((supplier) => {
      if (!query) return true

      return [supplier.name, supplier.phone, supplier.email, supplier.address]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    })
  }, [supplierSearch, suppliers])

  const sortedSuppliers = useMemo(() => sortByName(filteredSuppliers), [filteredSuppliers])

  const customerOverview = useMemo(
    () => ({
      count: sortedCustomers.length,
      receivable: sortedCustomers.reduce(
        (sum, customer) => sum + Number(customer.pending_balance || customer.pending_invoice_amount || 0),
        0
      ),
      advance: sortedCustomers.reduce((sum, customer) => sum + Number(customer.advance_balance || 0), 0),
    }),
    [sortedCustomers]
  )

  const supplierOverview = useMemo(
    () => ({
      totalBill: sortedSuppliers.reduce((sum, supplier) => sum + Number(supplier.total_bill || 0), 0),
      paid: sortedSuppliers.reduce((sum, supplier) => sum + Number(supplier.paid_amount || 0), 0),
      pending: sortedSuppliers.reduce((sum, supplier) => sum + Number(supplier.pending_amount || 0), 0),
    }),
    [sortedSuppliers]
  )

  const supplierBillOptions = useMemo(
    () =>
      sortByName(
        suppliers.filter((supplier) =>
          matchesSearchText(
            [supplier.name, supplier.phone, supplier.email, supplier.address, getSupplierLookupLabel(supplier)],
            supplierBillSearch
          )
        )
      ),
    [supplierBillSearch, suppliers]
  )

  const supplierPaymentSupplierOptions = useMemo(
    () =>
      sortByName(
        suppliers.filter((supplier) =>
          matchesSearchText(
            [supplier.name, supplier.phone, supplier.email, supplier.address, getSupplierLookupLabel(supplier)],
            supplierPaymentSupplierSearch
          )
        )
      ),
    [supplierPaymentSupplierSearch, suppliers]
  )

  const pendingSupplierInvoiceOptions = useMemo(() => {
    return [...supplierPaymentInvoices]
      .filter((invoice) =>
        matchesSearchText(
          [
            invoice.bill_number,
            invoice.supplier_name,
            invoice.invoice_date,
            invoice.pending_amount,
            invoice.status,
            getSupplierInvoiceLookupLabel(invoice),
          ],
          supplierPaymentInvoiceSearch
        )
      )
      .sort((leftInvoice, rightInvoice) =>
        String(rightInvoice?.invoice_date || "").localeCompare(String(leftInvoice?.invoice_date || ""))
      )
  }, [supplierPaymentInvoiceSearch, supplierPaymentInvoices])

  const customerLedgerEntries = customerLedger?.data || []
  const customerLedgerInvoices = customerLedger?.invoices || []
  const customerItemLedger = customerLedger?.item_ledger || []
  const customerLedgerTotals = useMemo(
    () =>
      customerLedgerEntries.reduce(
        (totals, entry) => ({
          debit: totals.debit + Number(entry.debit || 0),
          credit: totals.credit + Number(entry.credit || 0),
        }),
        { debit: 0, credit: 0 }
      ),
    [customerLedgerEntries]
  )
  const filteredCustomerLedgerEntries = useMemo(
    () =>
      customerLedgerEntries.filter((entry) =>
        matchesSearchText(
          [
            entry.date,
            entry.description,
            entry.mode,
            entry.type,
            entry.amount,
            entry.debit,
            entry.credit,
            entry.balance,
          ],
          customerDetailSearch
        )
      ),
    [customerDetailSearch, customerLedgerEntries]
  )
  const filteredCustomerInvoices = useMemo(
    () =>
      customerLedgerInvoices.filter((invoice) =>
        matchesSearchText(
          [
            invoice.invoice_number,
            invoice.created_at,
            invoice.payment_status,
            invoice.customer_name,
            invoice.total,
            invoice.paid_amount,
            invoice.balance_amount,
          ],
          customerDetailSearch
        )
      ),
    [customerDetailSearch, customerLedgerInvoices]
  )
  const filteredCustomerItemLedger = useMemo(
    () =>
      customerItemLedger.filter((item) =>
        matchesSearchText(
          [
            item.invoice_number,
            item.created_at,
            item.product_name,
            item.sku,
            item.variant_name,
            item.color,
            item.size,
            item.quantity,
            item.price,
            item.total,
          ],
          customerDetailSearch
        )
      ),
    [customerDetailSearch, customerItemLedger]
  )

  const supplierLedgerEntries = useMemo(() => {
    const rows = supplierLedger?.data || []

    return [...rows].sort((leftRow, rightRow) => {
      const leftValue = leftRow?.created_at || leftRow?.date || ""
      const rightValue = rightRow?.created_at || rightRow?.date || ""
      return String(rightValue).localeCompare(String(leftValue))
    })
  }, [supplierLedger])

  const supplierLedgerInvoices = supplierLedger?.invoices || []
  const filteredSupplierLedgerEntries = useMemo(
    () =>
      supplierLedgerEntries.filter((entry) =>
        matchesSearchText(
          [
            entry.date,
            entry.created_at,
            entry.mode,
            entry.description,
            entry.bill_number,
            entry.amount,
            entry.debit,
            entry.credit,
            entry.balance,
            getSupplierLedgerTypeLabel(entry.type),
          ],
          supplierLedgerSearch
        )
      ),
    [supplierLedgerEntries, supplierLedgerSearch]
  )

  const filteredSupplierInvoices = useMemo(
    () =>
      supplierLedgerInvoices.filter((invoice) =>
        matchesSearchText(
          [
            invoice.bill_number,
            invoice.invoice_date,
            invoice.status,
            invoice.total_amount,
            invoice.paid_amount,
            invoice.pending_amount,
          ],
          supplierLedgerSearch
        )
      ),
    [supplierLedgerInvoices, supplierLedgerSearch]
  )

  const selectedSupplierForBill =
    suppliers.find((supplier) => supplier.id === supplierBillSupplierId) || null

  const selectedSupplierPaymentInvoice =
    supplierPaymentInvoices.find((invoice) => invoice.id === supplierPaymentInvoiceId) || null

  const customerStats = [
    { key: "count", label: "Customers", value: String(customerOverview.count) },
    {
      key: "receivable",
      label: "To Receive",
      value: formatCurrency(customerOverview.receivable),
      tone: "warning",
    },
    {
      key: "advance",
      label: "Advance",
      value: formatCurrency(customerOverview.advance),
      tone: "success",
    },
  ]

  const supplierStats = [
    {
      key: "total_bill",
      label: "Total Bills",
      value: formatCurrency(supplierOverview.totalBill),
      tone: "default",
    },
    {
      key: "paid",
      label: "Paid",
      value: formatCurrency(supplierOverview.paid),
      tone: "success",
    },
    {
      key: "pending",
      label: "Pending",
      value: formatCurrency(supplierOverview.pending),
      tone: "warning",
    },
  ]

  return (
    <div className="min-h-screen bg-[#050816] text-slate-100">
      <div className="mx-auto max-w-6xl space-y-4 px-4 py-4 md:px-6">
        <div className="space-y-4">
          <div className="flex flex-col gap-3 md:flex-row md:items-center">
            <div className="relative flex-1">
              <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
              <Input
                value={activeTab === "suppliers" ? supplierSearch : customerSearch}
                onChange={(event) =>
                  activeTab === "suppliers"
                    ? setSupplierSearch(event.target.value)
                    : setCustomerSearch(event.target.value)
                }
                placeholder={activeTab === "suppliers" ? "Search supplier..." : "Search customer..."}
                className="h-12 rounded-xl border-white/8 bg-white/[0.02] pl-11 pr-4 text-white shadow-[0_14px_30px_rgba(2,6,23,0.08)] backdrop-blur-xl placeholder:text-slate-500"
              />
            </div>

            <div className="flex items-center gap-3 md:hidden">
              {activeTab === "customers" ? (
                <Select value={customerBalanceFilter} onValueChange={setCustomerBalanceFilter}>
                  <SelectTrigger className="h-12 w-12 rounded-xl border-white/8 bg-white/[0.02] px-0 text-white shadow-[0_14px_30px_rgba(2,6,23,0.08)] backdrop-blur-xl">
                    <SlidersHorizontal className="h-4 w-4" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Customers</SelectItem>
                    <SelectItem value="pending">Pending</SelectItem>
                    <SelectItem value="advance">Advance</SelectItem>
                  </SelectContent>
                </Select>
              ) : null}
            </div>

            <div className="hidden items-center gap-3 md:flex">
              <Button
                className="h-12 min-w-[205px] rounded-xl bg-indigo-500 px-6 text-white hover:bg-indigo-400"
                onClick={() => (activeTab === "suppliers" ? setOpenSupplier(true) : setOpenCustomer(true))}
              >
                <Plus className="mr-2 h-4 w-4" />
                {activeTab === "suppliers" ? "Add Supplier" : "Add Customer"}
              </Button>

              {activeTab === "customers" ? (
                <>
                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-300">Sort</span>
                    <Select value={customerSort} onValueChange={setCustomerSort}>
                      <SelectTrigger className="h-12 w-[180px] rounded-xl border-white/8 bg-white/[0.02] text-white shadow-[0_14px_30px_rgba(2,6,23,0.08)] backdrop-blur-xl">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="az">A-Z</SelectItem>
                        <SelectItem value="za">Z-A</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center gap-2">
                    <span className="text-sm text-slate-300">Filter</span>
                    <Select value={customerBalanceFilter} onValueChange={setCustomerBalanceFilter}>
                      <SelectTrigger className="h-12 w-[180px] rounded-xl border-white/8 bg-white/[0.02] text-white shadow-[0_14px_30px_rgba(2,6,23,0.08)] backdrop-blur-xl">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Customers</SelectItem>
                        <SelectItem value="pending">Pending</SelectItem>
                        <SelectItem value="advance">Advance</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </>
              ) : null}
            </div>
          </div>

          <StatsCardGrid
            items={activeTab === "suppliers" ? supplierStats : customerStats}
            className="grid-cols-3 md:grid-cols-3 lg:grid-cols-3"
          />

          {activeTab === "suppliers" ? (
            <div className="grid grid-cols-2 gap-3 md:hidden">
              <ActionFeatureCard
                icon={<Plus className="h-4 w-4" />}
                badge="New"
                title="Add Bill"
                description="Record a new supplier bill and update payable balance together."
                actionLabel="Open Bill Form"
                tone="indigo"
                onClick={() => openSupplierBillDialogFor()}
              />

              <ActionFeatureCard
                icon={<Wallet className="h-4 w-4" />}
                badge="Pending"
                title="Pay Bill"
                description="Clear pending supplier bills directly from the payment ledger."
                actionLabel="Open Payment Form"
                tone="emerald"
                onClick={() => openSupplierPaymentDialogFor()}
              />
            </div>
          ) : null}
        </div>

        {activeTab === "customers" ? (
          <div className="space-y-3">
            {sortedCustomers.length === 0 ? (
              <div className={`${floatingPanelClass} px-6 py-12 text-center text-sm text-slate-400`}>
                No customers found
              </div>
            ) : (
              <>
                <div className="space-y-3 md:hidden">
                  {sortedCustomers.map((customer) => (
                    <EntityCard
                      key={customer.id}
                      compact
                      icon={<User className="h-4 w-4" />}
                      title={customer.name}
                      subtitle={customer.phone || "No phone added"}
                      metaLines={[
                        `${Number(customer.total_invoices || 0)} invoice${Number(customer.total_invoices || 0) === 1 ? "" : "s"}`,
                        getCustomerBalanceText(customer),
                      ]}
                      amount={formatCurrency(customer.total_bill)}
                      amountTone="success"
                      status={getCustomerStatus(customer)}
                      breakdown={[
                        { key: "bill", label: "Bill", value: formatCurrency(customer.total_bill) },
                        {
                          key: "pending",
                          label: "Pending",
                          value: formatCurrency(customer.pending_balance || customer.pending_invoice_amount),
                          tone: "warning",
                        },
                        {
                          key: "advance",
                          label: "Advance",
                          value: formatCurrency(customer.advance_balance),
                          tone: "success",
                        },
                      ]}
                      actions={[
                        {
                          key: "view",
                          label: "View",
                          onClick: () => openCustomerDetails(customer),
                        },
                        {
                          key: "invoice",
                          label: "Invoice",
                          icon: FileText,
                          onClick: () => handleCreateInvoice(customer),
                        },
                        {
                          key: "advance",
                          label: "Adv",
                          icon: Wallet,
                          onClick: () => openAdvanceDialogFor(customer),
                          disabled: !customer.can_add_advance,
                        },
                        {
                          key: "edit",
                          label: "Edit",
                          icon: Edit,
                          onClick: () => handleEditCustomer(customer),
                        },
                        {
                          key: "delete",
                          label: "Delete",
                          icon: Trash2,
                          tone: "danger",
                          onClick: () => deleteCustomer(customer),
                        },
                      ]}
                      className={floatingCardClass}
                    />
                  ))}
                </div>

                <div className={`${desktopTableShellClass} hidden md:block`}>
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 bg-white/[0.03] hover:bg-white/[0.03]">
                        <TableHead className="w-10 text-slate-300">#</TableHead>
                        <TableHead className="text-slate-300">Name</TableHead>
                        <TableHead className="text-slate-300">Email</TableHead>
                        <TableHead className="text-slate-300">Phone</TableHead>
                        <TableHead className="text-slate-300">Invoices</TableHead>
                        <TableHead className="text-right text-slate-300">Total Billing</TableHead>
                        <TableHead className="text-slate-300">Balance</TableHead>
                        <TableHead className="text-slate-300">Address</TableHead>
                        <TableHead className="text-right text-slate-300">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedCustomers.map((customer, index) => (
                        <TableRow
                          key={customer.id}
                          className="cursor-pointer border-white/10 text-slate-100 hover:bg-white/[0.025]"
                          onClick={() => openCustomerDetails(customer)}
                        >
                          <TableCell className="text-slate-200">{index + 1}</TableCell>
                          <TableCell className="font-medium text-white">{customer.name}</TableCell>
                          <TableCell className="text-slate-300">{customer.email || "-"}</TableCell>
                          <TableCell className="text-slate-300">{customer.phone || "-"}</TableCell>
                          <TableCell className="text-slate-300">{Number(customer.total_invoices || 0)}</TableCell>
                          <TableCell className="whitespace-nowrap text-right font-semibold text-emerald-300">
                            {formatCurrency(customer.total_bill)}
                          </TableCell>
                          <TableCell className="text-slate-300">
                            {Number(customer.pending_balance || customer.pending_invoice_amount || 0) > 0 ? (
                              <span className="font-medium text-amber-200">
                                Pending {formatCurrency(customer.pending_balance || customer.pending_invoice_amount)}
                              </span>
                            ) : Number(customer.advance_balance || 0) > 0 ? (
                              <span className="font-medium text-emerald-200">
                                Advance {formatCurrency(customer.advance_balance)}
                              </span>
                            ) : (
                              "No pending or advance"
                            )}
                          </TableCell>
                          <TableCell className="max-w-[200px] truncate text-slate-300">{customer.address || "-"}</TableCell>
                          <TableCell className="text-right" onClick={(event) => event.stopPropagation()}>
                            <div className="flex justify-end gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => handleCreateInvoice(customer)}
                              >
                                Invoice
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => openAdvanceDialogFor(customer)}
                                disabled={!customer.can_add_advance}
                              >
                                Adv
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => handleEditCustomer(customer)}
                              >
                                Edit
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-rose-400/20 bg-rose-500/10 px-3 text-xs text-rose-200 hover:bg-rose-500/20 hover:text-white"
                                onClick={() => deleteCustomer(customer)}
                              >
                                Delete
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </div>
        ) : (
          <div className="space-y-3">
            {sortedSuppliers.length === 0 ? (
              <div className={`${floatingPanelClass} px-6 py-12 text-center text-sm text-slate-400`}>
                No suppliers found
              </div>
            ) : (
              <>
                <div className="space-y-3 md:hidden">
                  {sortedSuppliers.map((supplier) => (
                    <EntityCard
                      key={supplier.id}
                      icon={<Building2 className="h-4 w-4" />}
                      title={supplier.name}
                      subtitle={supplier.phone || supplier.email || "No contact added"}
                      metaLines={supplier.address ? [supplier.address] : []}
                      amount={formatCurrency(supplier.pending_amount)}
                      amountTone={Number(supplier.pending_amount || 0) > 0 ? "warning" : "success"}
                      status={getSupplierStatus(supplier)}
                      breakdown={[
                        { key: "total_bill", label: "Bill", value: formatCurrency(supplier.total_bill) },
                        {
                          key: "paid",
                          label: "Paid",
                          value: formatCurrency(supplier.paid_amount),
                          tone: "success",
                        },
                        {
                          key: "pending",
                          label: "Pending",
                          value: formatCurrency(supplier.pending_amount),
                          tone: "warning",
                        },
                      ]}
                      actions={[
                        {
                          key: "bill",
                          label: "Add Bill",
                          icon: Plus,
                          onClick: () => openSupplierBillDialogFor(supplier),
                        },
                        {
                          key: "pay",
                          label: "Pay",
                          icon: Wallet,
                          onClick: () => openSupplierPaymentDialogFor(supplier),
                          disabled: Number(supplier.pending_amount || 0) <= 0,
                        },
                        {
                          key: "ledger",
                          label: "Ledger",
                          icon: FileText,
                          onClick: () => openSupplierDetails(supplier),
                        },
                      ]}
                      className={floatingCardClass}
                    />
                  ))}
                </div>

                <div className={`${desktopTableShellClass} hidden md:block`}>
                  <Table>
                    <TableHeader>
                      <TableRow className="border-white/10 bg-white/[0.03] hover:bg-white/[0.03]">
                        <TableHead className="w-10 text-slate-300">#</TableHead>
                        <TableHead className="text-slate-300">Supplier</TableHead>
                        <TableHead className="text-slate-300">Contact</TableHead>
                        <TableHead className="text-right text-slate-300">Total Bill</TableHead>
                        <TableHead className="text-right text-slate-300">Paid</TableHead>
                        <TableHead className="text-right text-slate-300">Pending</TableHead>
                        <TableHead className="text-right text-slate-300">Balance</TableHead>
                        <TableHead className="text-slate-300">Address</TableHead>
                        <TableHead className="text-right text-slate-300">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sortedSuppliers.map((supplier, index) => (
                        <TableRow
                          key={supplier.id}
                          className="border-white/10 text-slate-100 hover:bg-white/[0.025]"
                        >
                          <TableCell className="text-slate-200">{index + 1}</TableCell>
                          <TableCell className="font-medium text-white">
                            <div className="space-y-0.5">
                              <p>{supplier.name}</p>
                              <p className="max-w-[220px] truncate text-xs text-slate-400">{supplier.email || "-"}</p>
                            </div>
                          </TableCell>
                          <TableCell className="text-slate-300">{supplier.phone || "-"}</TableCell>
                          <TableCell className="whitespace-nowrap text-right font-semibold text-white">
                            {formatCurrency(supplier.total_bill)}
                          </TableCell>
                          <TableCell className="whitespace-nowrap text-right font-semibold text-emerald-300">
                            {formatCurrency(supplier.paid_amount)}
                          </TableCell>
                          <TableCell className="whitespace-nowrap text-right font-semibold text-amber-200">
                            {formatCurrency(supplier.pending_amount)}
                          </TableCell>
                          <TableCell className={`whitespace-nowrap text-right font-semibold ${
                            Number(supplier.current_balance || 0) > 0 ? "text-rose-300" : Number(supplier.current_balance || 0) < 0 ? "text-emerald-300" : "text-white"
                          }`}>
                            {formatCurrency(supplier.current_balance)}
                          </TableCell>
                          <TableCell className="max-w-[180px] truncate text-slate-300">{supplier.address || "-"}</TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-2">
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => openSupplierBillDialogFor(supplier)}
                              >
                                Add Bill
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => openSupplierPaymentDialogFor(supplier)}
                                disabled={Number(supplier.pending_amount || 0) <= 0}
                              >
                                Pay Bill
                              </Button>
                              <Button
                                size="sm"
                                variant="outline"
                                className="h-8 rounded-xl border-white/10 bg-transparent px-3 text-xs text-white hover:bg-white/5 hover:text-white"
                                onClick={() => openSupplierDetails(supplier)}
                              >
                                Ledger Entry
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      <Dialog
        open={openCustomer}
        onOpenChange={(nextOpen) => {
          setOpenCustomer(nextOpen)
          if (!nextOpen) resetCustomerForm()
        }}
      >
        <DialogContent className="max-w-[95vw] rounded-3xl border-white/10 bg-[#060b1a]/95 text-slate-100 backdrop-blur-xl sm:max-w-[520px]">
          <DialogHeader>
            <DialogTitle>{editingCustomer ? "Edit Customer" : "Add Customer"}</DialogTitle>
          </DialogHeader>

          <form onSubmit={submitCustomer} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="customer-name">Full Name</Label>
              <Input
                id="customer-name"
                value={customerForm.name}
                onChange={(event) => setCustomerForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Enter customer name"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="customer-email">Email</Label>
              <Input
                id="customer-email"
                type="email"
                value={customerForm.email}
                onChange={(event) => setCustomerForm((current) => ({ ...current, email: event.target.value }))}
                placeholder="customer@example.com"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="customer-phone">Phone Number</Label>
              <Input
                id="customer-phone"
                value={customerForm.phone}
                onChange={(event) => setCustomerForm((current) => ({ ...current, phone: event.target.value }))}
                placeholder="+91 98765 43210"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="customer-address">Address</Label>
              <Input
                id="customer-address"
                value={customerForm.address}
                onChange={(event) => setCustomerForm((current) => ({ ...current, address: event.target.value }))}
                placeholder="Street address, city, state"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              {editingCustomer ? (
                <Button type="button" variant="outline" className="h-12 rounded-2xl" onClick={resetCustomerForm}>
                  Cancel Edit
                </Button>
              ) : null}
              <Button type="submit" className="h-12 rounded-2xl">
                {editingCustomer ? "Update Customer" : "Add Customer"}
              </Button>
            </div>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={openSupplier}
        onOpenChange={(nextOpen) => {
          setOpenSupplier(nextOpen)
          if (!nextOpen) resetSupplierForm()
        }}
      >
        <DialogContent className="max-w-[95vw] rounded-3xl border-white/10 bg-[#060b1a]/95 text-slate-100 backdrop-blur-xl sm:max-w-[520px]">
          <DialogHeader>
            <DialogTitle>Add Supplier</DialogTitle>
          </DialogHeader>

          <form onSubmit={submitSupplier} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="supplier-name">Supplier Name</Label>
              <Input
                id="supplier-name"
                value={supplierForm.name}
                onChange={(event) => setSupplierForm((current) => ({ ...current, name: event.target.value }))}
                placeholder="Enter supplier name"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                required
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="supplier-phone">Phone Number</Label>
              <Input
                id="supplier-phone"
                value={supplierForm.phone}
                onChange={(event) => setSupplierForm((current) => ({ ...current, phone: event.target.value }))}
                placeholder="+91 98765 43210"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="supplier-email">Email</Label>
              <Input
                id="supplier-email"
                type="email"
                value={supplierForm.email}
                onChange={(event) => setSupplierForm((current) => ({ ...current, email: event.target.value }))}
                placeholder="supplier@example.com"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="supplier-address">Address</Label>
              <Input
                id="supplier-address"
                value={supplierForm.address}
                onChange={(event) => setSupplierForm((current) => ({ ...current, address: event.target.value }))}
                placeholder="Street address, city, state"
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
              />
            </div>

            <Button type="submit" className="h-12 w-full rounded-2xl">
              Add Supplier
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={advanceDialogOpen}
        onOpenChange={(nextOpen) => {
          setAdvanceDialogOpen(nextOpen)
          if (!nextOpen) {
            setAdvanceCustomer(null)
            setAdvanceForm(createEmptyAdvanceForm())
          }
        }}
      >
        <DialogContent className="max-w-[95vw] rounded-3xl border-white/10 bg-[#060b1a]/95 text-slate-100 backdrop-blur-xl sm:max-w-[520px]">
          <DialogHeader>
            <DialogTitle>Add Advance</DialogTitle>
          </DialogHeader>

          <form onSubmit={submitAdvance} className="space-y-4">
            <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-sm text-slate-300">
              {advanceCustomer?.name || "Selected customer"}
            </div>

            <div className="space-y-2">
              <Label htmlFor="advance-amount">Amount</Label>
              <Input
                id="advance-amount"
                type="number"
                min="0"
                value={advanceForm.amount}
                onChange={(event) => setAdvanceForm((current) => ({ ...current, amount: event.target.value }))}
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                placeholder="0.00"
              />
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label>Payment Mode</Label>
                <Select
                  value={advanceForm.payment_mode}
                  onValueChange={(value) => setAdvanceForm((current) => ({ ...current, payment_mode: value }))}
                >
                  <SelectTrigger className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white">
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

              <div className="space-y-2">
                <Label htmlFor="advance-date">Date</Label>
                <Input
                  id="advance-date"
                  type="date"
                  value={advanceForm.payment_date}
                  onChange={(event) => setAdvanceForm((current) => ({ ...current, payment_date: event.target.value }))}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="advance-reference">Reference</Label>
              <Input
                id="advance-reference"
                value={advanceForm.reference}
                onChange={(event) => setAdvanceForm((current) => ({ ...current, reference: event.target.value }))}
                className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                placeholder="Optional reference"
              />
            </div>

            <Button type="submit" className="h-12 w-full rounded-2xl" disabled={advanceSubmitting}>
              {advanceSubmitting ? "Saving..." : "Add Advance"}
            </Button>
          </form>
        </DialogContent>
      </Dialog>

      <Dialog
        open={supplierBillDialogOpen}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) resetSupplierBillDialog()
        }}
      >
        <DialogContent className="max-w-[95vw] rounded-3xl border-white/10 bg-[#060b1a]/95 text-slate-100 backdrop-blur-xl sm:max-w-[520px]">
          <DialogHeader>
            <DialogTitle>Add Supplier Bill</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Select Supplier</Label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  value={supplierBillSearch}
                  onChange={(event) => {
                    setSupplierBillSearch(event.target.value)
                    setSupplierBillDropdownOpen(true)
                  }}
                  onFocus={() => setSupplierBillDropdownOpen(true)}
                  placeholder="Search supplier..."
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 pl-11 text-white"
                />
                {supplierBillDropdownOpen ? (
                  <div className="absolute inset-x-0 top-[calc(100%+0.5rem)] z-30 max-h-56 overflow-y-auto rounded-2xl border border-white/10 bg-[#081121] p-2 shadow-[0_18px_36px_rgba(15,23,42,0.28)]">
                    {supplierBillOptions.length ? (
                      supplierBillOptions.map((supplier) => (
                        <button
                          key={supplier.id}
                          type="button"
                          className="flex w-full items-start rounded-xl px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/5"
                          onClick={() => {
                            setSupplierBillSupplierId(supplier.id)
                            setSupplierBillSearch(getSupplierLookupLabel(supplier))
                            setSupplierBillDropdownOpen(false)
                          }}
                        >
                          <div>
                            <p className="font-medium text-white">{supplier.name}</p>
                            <p className="text-xs text-slate-400">{getSupplierLookupLabel(supplier)}</p>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="px-3 py-2 text-sm text-slate-400">No supplier found</div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="supplier-bill-amount">Bill Amount</Label>
                <Input
                  id="supplier-bill-amount"
                  type="number"
                  min="0"
                  value={supplierBillAmount}
                  onChange={(event) => setSupplierBillAmount(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                  placeholder="0.00"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="supplier-bill-date">Bill Date</Label>
                <Input
                  id="supplier-bill-date"
                  type="date"
                  value={supplierBillDate}
                  onChange={(event) => setSupplierBillDate(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                />
              </div>
            </div>

            {selectedSupplierForBill ? (
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-sm text-slate-300">
                {selectedSupplierForBill.name}
              </div>
            ) : null}

            <Button className="h-12 w-full rounded-2xl" onClick={addSupplierBill}>
              Save Bill
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={supplierPaymentDialogOpen}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) resetSupplierPaymentDialog()
        }}
      >
        <DialogContent className="max-w-[95vw] rounded-3xl border-white/10 bg-[#060b1a]/95 text-slate-100 backdrop-blur-xl sm:max-w-[560px]">
          <DialogHeader>
            <DialogTitle>Pay Supplier Bill</DialogTitle>
          </DialogHeader>

          <div className="space-y-4">
            <div className="space-y-2">
              <Label>Select Supplier</Label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  value={supplierPaymentSupplierSearch}
                  onChange={(event) => {
                    setSupplierPaymentSupplierSearch(event.target.value)
                    setSupplierPaymentSupplierDropdownOpen(true)
                  }}
                  onFocus={() => setSupplierPaymentSupplierDropdownOpen(true)}
                  placeholder="Search supplier..."
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 pl-11 text-white"
                />
                {supplierPaymentSupplierDropdownOpen ? (
                  <div className="absolute inset-x-0 top-[calc(100%+0.5rem)] z-30 max-h-56 overflow-y-auto rounded-2xl border border-white/10 bg-[#081121] p-2 shadow-[0_18px_36px_rgba(15,23,42,0.28)]">
                    {supplierPaymentSupplierOptions.length ? (
                      supplierPaymentSupplierOptions.map((supplier) => (
                        <button
                          key={supplier.id}
                          type="button"
                          className="flex w-full items-start rounded-xl px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/5"
                          onClick={() => selectSupplierForPayment(supplier)}
                        >
                          <div>
                            <p className="font-medium text-white">{supplier.name}</p>
                            <p className="text-xs text-slate-400">{getSupplierLookupLabel(supplier)}</p>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="px-3 py-2 text-sm text-slate-400">No supplier found</div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>

            <div className="space-y-2">
              <Label>Select Pending Bill</Label>
              <div className="relative">
                <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <Input
                  value={supplierPaymentInvoiceSearch}
                  onChange={(event) => {
                    setSupplierPaymentInvoiceSearch(event.target.value)
                    setSupplierPaymentInvoiceDropdownOpen(true)
                  }}
                  onFocus={() => setSupplierPaymentInvoiceDropdownOpen(true)}
                  placeholder={supplierPaymentSupplierId ? "Search pending bill..." : "Select supplier first"}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 pl-11 text-white"
                  disabled={!supplierPaymentSupplierId}
                />
                {supplierPaymentInvoiceDropdownOpen && supplierPaymentSupplierId ? (
                  <div className="absolute inset-x-0 top-[calc(100%+0.5rem)] z-30 max-h-56 overflow-y-auto rounded-2xl border border-white/10 bg-[#081121] p-2 shadow-[0_18px_36px_rgba(15,23,42,0.28)]">
                    {pendingSupplierInvoiceOptions.length ? (
                      pendingSupplierInvoiceOptions.map((invoice) => (
                        <button
                          key={invoice.id}
                          type="button"
                          className="flex w-full items-start rounded-xl px-3 py-2 text-left text-sm text-slate-200 hover:bg-white/5"
                          onClick={() => {
                            setSupplierPaymentInvoiceId(invoice.id)
                            setSupplierPaymentInvoiceSearch(getSupplierInvoiceLookupLabel(invoice))
                            setSupplierPaymentAmount(String(Number(invoice.pending_amount || 0)))
                            setSupplierPaymentInvoiceDropdownOpen(false)
                          }}
                        >
                          <div>
                            <p className="font-medium text-white">{invoice.bill_number || invoice.id}</p>
                            <p className="text-xs text-slate-400">{getSupplierInvoiceLookupLabel(invoice)}</p>
                          </div>
                        </button>
                      ))
                    ) : (
                      <div className="px-3 py-2 text-sm text-slate-400">No pending bill found</div>
                    )}
                  </div>
                ) : null}
              </div>
            </div>

            {supplierPaymentLoading ? (
              <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] px-4 py-6 text-center text-sm text-slate-400">
                Loading pending bills...
              </div>
            ) : null}

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="supplier-payment-amount">Amount to Pay</Label>
                <Input
                  id="supplier-payment-amount"
                  type="number"
                  min="0"
                  value={supplierPaymentAmount}
                  onChange={(event) => setSupplierPaymentAmount(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                  placeholder="0.00"
                />
              </div>

              <div className="space-y-2">
                <Label>Payment Mode</Label>
                <Select value={supplierPaymentMode} onValueChange={setSupplierPaymentMode}>
                  <SelectTrigger className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white">
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

            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="supplier-payment-date">Payment Date</Label>
                <Input
                  id="supplier-payment-date"
                  type="date"
                  value={supplierPaymentDate}
                  onChange={(event) => setSupplierPaymentDate(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="supplier-payment-notes">Notes</Label>
                <Input
                  id="supplier-payment-notes"
                  value={supplierPaymentNotes}
                  onChange={(event) => setSupplierPaymentNotes(event.target.value)}
                  className="h-12 rounded-2xl border-white/10 bg-slate-950/50 text-white"
                  placeholder="Optional notes"
                />
              </div>
            </div>

            {selectedSupplierPaymentInvoice ? (
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3 text-sm text-slate-300">
                Pending: {formatCurrency(selectedSupplierPaymentInvoice.pending_amount)}
              </div>
            ) : null}

            <Button className="h-12 w-full rounded-2xl" onClick={paySupplierBill}>
              Pay Bill
            </Button>
          </div>
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(viewCustomer)}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
            setViewCustomer(null)
            setCustomerLedger(null)
          }
        }}
      >
        <DialogContent className="max-h-[92vh] max-w-[95vw] overflow-hidden rounded-3xl border-white/10 bg-[#060b1a]/95 p-0 text-slate-100 backdrop-blur-xl sm:max-w-5xl">
          <DialogHeader className="border-b border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.72),rgba(15,23,42,0.38))] px-5 py-4">
            <DialogTitle>Customer Details</DialogTitle>
          </DialogHeader>

          {viewCustomer ? (
            <div className="max-h-[78vh] overflow-y-auto px-4 py-4 sm:px-5">
              <div className="space-y-4">
                <div className={`${floatingPanelClass} p-4`}>
                  <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                    <div className="space-y-1">
                      <p className="text-lg font-semibold text-white">{viewCustomer.name}</p>
                      {viewCustomer.phone ? (
                        <p className="text-sm text-slate-300">{viewCustomer.phone}</p>
                      ) : null}
                      {viewCustomer.email ? (
                        <p className="text-sm text-slate-400 break-all">{viewCustomer.email}</p>
                      ) : null}
                      {viewCustomer.address ? (
                        <p className="text-sm text-slate-400 break-words">{viewCustomer.address}</p>
                      ) : null}
                      <p className="pt-1 text-sm font-medium text-slate-300">{getCustomerBalanceText(viewCustomer)}</p>
                    </div>

                    <div className="grid grid-cols-2 gap-2 md:min-w-[22rem]">
                      <ModalStat label="Invoices" value={String(viewCustomer.total_invoices || customerLedgerInvoices.length)} />
                      <ModalStat label="Total Sale" value={formatCurrency(viewCustomer.total_bill)} tone="success" />
                      <ModalStat
                        label="Total Balance"
                        value={formatCurrency(viewCustomer.current_balance || viewCustomer.pending_balance || viewCustomer.pending_invoice_amount)}
                        tone={Number(viewCustomer.current_balance || viewCustomer.pending_balance || viewCustomer.pending_invoice_amount || 0) > 0 ? "warning" : "default"}
                      />
                      <ModalStat label="Pending / Advance" value={getCustomerBalanceSummary(viewCustomer)} tone="default" />
                    </div>
                  </div>
                </div>

                {customerLedgerLoading ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.04] p-8 text-center text-slate-400">
                    Loading customer ledger...
                  </div>
                ) : (
                  <div className={`${floatingPanelClass} p-4`}>
                    <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                      <div className="flex flex-wrap gap-3">
                        <Button
                          variant={customerDetailTab === "ledger" ? "default" : "outline"}
                          className={`h-11 rounded-2xl ${
                            customerDetailTab === "ledger"
                              ? "bg-slate-900 text-white hover:bg-slate-800"
                              : "border-white/10 bg-slate-950/50 text-slate-300 hover:bg-white/5 hover:text-white"
                          }`}
                          onClick={() => setCustomerDetailTab("ledger")}
                        >
                          Ledger
                        </Button>
                        <Button
                          variant={customerDetailTab === "invoices" ? "default" : "outline"}
                          className={`h-11 rounded-2xl ${
                            customerDetailTab === "invoices"
                              ? "bg-slate-900 text-white hover:bg-slate-800"
                              : "border-white/10 bg-slate-950/50 text-slate-300 hover:bg-white/5 hover:text-white"
                          }`}
                          onClick={() => setCustomerDetailTab("invoices")}
                        >
                          Invoices
                        </Button>
                        <Button
                          variant={customerDetailTab === "items" ? "default" : "outline"}
                          className={`h-11 rounded-2xl ${
                            customerDetailTab === "items"
                              ? "bg-slate-900 text-white hover:bg-slate-800"
                              : "border-white/10 bg-slate-950/50 text-slate-300 hover:bg-white/5 hover:text-white"
                          }`}
                          onClick={() => setCustomerDetailTab("items")}
                        >
                          Item Ledger
                        </Button>
                      </div>

                      <div className="relative lg:w-[28rem]">
                        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                        <Input
                          value={customerDetailSearch}
                          onChange={(event) => setCustomerDetailSearch(event.target.value)}
                          placeholder={
                            customerDetailTab === "ledger"
                              ? "Search ledger..."
                              : customerDetailTab === "invoices"
                                ? "Search invoices..."
                                : "Search item ledger..."
                          }
                          className="h-11 rounded-2xl border-white/10 bg-slate-950/60 pl-9 text-white placeholder:text-slate-500"
                        />
                      </div>
                    </div>

                    {customerDetailTab === "ledger" ? (
                      filteredCustomerLedgerEntries.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-8 text-center text-slate-400">
                          No ledger entries found
                        </div>
                      ) : (
                        <div className="max-h-[24rem] overflow-auto rounded-2xl border border-white/10 bg-slate-950/40">
                          <Table>
                            <TableHeader>
                              <TableRow className="border-white/10 bg-white/[0.04]">
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
                              {filteredCustomerLedgerEntries.map((entry) => (
                                <TableRow key={entry.id} className="border-white/10">
                                  <TableCell className="whitespace-nowrap">{formatDateValue(entry.date)}</TableCell>
                                  <TableCell className="font-medium">{getCustomerLedgerTypeLabel(entry.type)}</TableCell>
                                  <TableCell>{String(entry.mode || "-").toUpperCase()}</TableCell>
                                  <TableCell className="max-w-[360px] truncate text-slate-300">{entry.description || "-"}</TableCell>
                                  <TableCell className="whitespace-nowrap text-right font-semibold text-white">
                                    {formatCurrency(entry.amount)}
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right text-amber-200">
                                    {Number(entry.debit || 0) > 0 ? formatCurrency(entry.debit) : "-"}
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right text-emerald-300">
                                    {Number(entry.credit || 0) > 0 ? formatCurrency(entry.credit) : "-"}
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right text-white">
                                    {formatCurrency(entry.display_balance ?? entry.balance)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      )
                    ) : customerDetailTab === "invoices" ? (
                      filteredCustomerInvoices.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-8 text-center text-slate-400">
                          No invoices found
                        </div>
                      ) : (
                        <div className="max-h-[24rem] overflow-auto rounded-2xl border border-white/10 bg-slate-950/40">
                          <Table>
                            <TableHeader>
                              <TableRow className="border-white/10 bg-white/[0.04]">
                                <TableHead>Invoice</TableHead>
                                <TableHead>Date</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead className="text-right">Total</TableHead>
                                <TableHead className="text-right">Paid</TableHead>
                                <TableHead className="text-right">Balance</TableHead>
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {filteredCustomerInvoices.map((invoice) => (
                                <TableRow key={invoice.id} className="border-white/10">
                                  <TableCell className="font-medium text-white">{invoice.invoice_number || invoice.id}</TableCell>
                                  <TableCell className="whitespace-nowrap text-slate-300">{formatDateValue(invoice.created_at)}</TableCell>
                                  <TableCell>
                                    <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold ${getInvoiceStatusClass(invoice.payment_status)}`}>
                                      {String(invoice.payment_status || "pending").toUpperCase()}
                                    </span>
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right font-semibold text-white">
                                    {formatCurrency(invoice.total)}
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right text-emerald-300">
                                    {formatCurrency(invoice.paid_amount)}
                                  </TableCell>
                                  <TableCell className="whitespace-nowrap text-right text-amber-200">
                                    {formatCurrency(invoice.balance_amount)}
                                  </TableCell>
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                      )
                    ) : filteredCustomerItemLedger.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.03] p-8 text-center text-slate-400">
                        No item ledger found
                      </div>
                    ) : (
                      <div className="max-h-[24rem] overflow-auto rounded-2xl border border-white/10 bg-slate-950/40">
                        <Table>
                          <TableHeader>
                            <TableRow className="border-white/10 bg-white/[0.04]">
                              <TableHead>Date</TableHead>
                              <TableHead>Invoice</TableHead>
                              <TableHead>Product</TableHead>
                              <TableHead>SKU</TableHead>
                              <TableHead className="text-right">Qty</TableHead>
                              <TableHead className="text-right">Price</TableHead>
                              <TableHead className="text-right">Total</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {filteredCustomerItemLedger.map((item) => (
                              <TableRow key={item.id} className="border-white/10">
                                <TableCell className="whitespace-nowrap text-slate-300">{formatDateValue(item.created_at)}</TableCell>
                                <TableCell className="font-medium text-white">{item.invoice_number || item.invoice_id}</TableCell>
                                <TableCell className="text-slate-300">{item.product_name || "-"}</TableCell>
                                <TableCell className="font-mono text-slate-300">{item.sku || "-"}</TableCell>
                                <TableCell className="whitespace-nowrap text-right text-white">{Number(item.quantity || 0)}</TableCell>
                                <TableCell className="whitespace-nowrap text-right text-slate-200">{formatCurrency(item.price)}</TableCell>
                                <TableCell className="whitespace-nowrap text-right font-semibold text-white">{formatCurrency(item.total)}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </div>
                    )}

                    <div className="mt-4 grid grid-cols-1 gap-3 sm:grid-cols-3">
                      <Button
                        variant="outline"
                        className="rounded-2xl"
                        onClick={() => {
                          setViewCustomer(null)
                          setCustomerLedger(null)
                        }}
                      >
                        Close
                      </Button>
                      <Button
                        variant="outline"
                        className="rounded-2xl"
                        onClick={() => openAdvanceDialogFor(viewCustomer)}
                        disabled={!viewCustomer.can_add_advance}
                      >
                        Add Advance
                      </Button>
                      <Button className="rounded-2xl" onClick={() => handleCreateInvoice(viewCustomer)}>
                        Create Invoice
                      </Button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      <Dialog
        open={Boolean(viewSupplierDetails)}
        onOpenChange={(nextOpen) => {
          if (!nextOpen) {
            setViewSupplierDetails(null)
            setSupplierLedger(null)
            setSupplierLedgerSearch("")
            setSupplierDetailTab("ledger")
          }
        }}
      >
        <DialogContent className="max-h-[92vh] max-w-[95vw] overflow-hidden rounded-3xl border-white/10 bg-[#060b1a]/95 p-0 text-slate-100 backdrop-blur-xl sm:max-w-4xl">
          <DialogHeader className="border-b border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.72),rgba(15,23,42,0.38))] px-5 py-4">
            <DialogTitle>Supplier Details</DialogTitle>
          </DialogHeader>

          {viewSupplierDetails ? (
            <div className="max-h-[78vh] overflow-y-auto px-4 py-4 sm:px-5">
              <div className="space-y-4">
                <div className={`${floatingPanelClass} p-4`}>
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
                    <div className="space-y-1.5">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="text-[10px] uppercase tracking-[0.24em] text-slate-400">Supplier Ledger</p>
                        <span className={`rounded-full px-2.5 py-0.5 text-[10px] font-semibold ${Number(viewSupplierDetails.pending_amount || 0) > 0 ? "bg-amber-400/15 text-amber-200" : "bg-emerald-400/15 text-emerald-200"}`}>
                          {Number(viewSupplierDetails.pending_amount || 0) > 0 ? "Pending" : "Settled"}
                        </span>
                      </div>
                      <p className="text-[15px] font-semibold leading-tight text-white sm:text-base">{viewSupplierDetails.name}</p>
                      {viewSupplierDetails.phone ? (
                        <p className="text-[12px] text-slate-300">{viewSupplierDetails.phone}</p>
                      ) : null}
                      {viewSupplierDetails.address ? (
                        <p className="text-[11px] leading-4 text-slate-400">{viewSupplierDetails.address}</p>
                      ) : null}
                    </div>

                    <div className="space-y-2">
                      <p className="text-right text-xs text-slate-400">{formatDateValue(viewSupplierDetails.created_at)}</p>
                      <div className="grid grid-cols-2 gap-2 lg:max-w-[24rem]">
                        <ModalStat label="Total Bill" value={formatCurrency(viewSupplierDetails.total_bill)} />
                        <ModalStat label="Paid" value={formatCurrency(viewSupplierDetails.paid_amount)} tone="success" />
                        <ModalStat label="Pending" value={formatCurrency(viewSupplierDetails.pending_amount)} tone="warning" />
                        <ModalStat
                          label="Ledger Balance"
                          value={formatCurrency(viewSupplierDetails.computed_balance ?? supplierLedger?.supplier?.computed_balance)}
                          tone={Number((viewSupplierDetails.computed_balance ?? supplierLedger?.supplier?.computed_balance) || 0) > 0 ? "warning" : "success"}
                        />
                      </div>
                    </div>
                  </div>

                  <div className="mt-3 flex gap-2">
                    <Button
                      variant="secondary"
                      className="h-8 rounded-2xl px-3 text-[11px]"
                      onClick={() => openSupplierBillDialogFor(viewSupplierDetails)}
                    >
                      <Plus className="mr-1.5 h-3.5 w-3.5" />
                      Add Bill
                    </Button>
                    <Button
                      variant="outline"
                      className="h-8 rounded-2xl border-white/20 bg-white/10 px-3 text-[11px] text-white hover:bg-white/15 hover:text-white"
                      onClick={() => openSupplierPaymentDialogFor(viewSupplierDetails)}
                      disabled={Number(viewSupplierDetails.pending_amount || 0) <= 0}
                    >
                      <Wallet className="mr-1.5 h-3.5 w-3.5" />
                      Pay Bill
                    </Button>
                  </div>
                </div>

                {supplierLedgerLoading ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.04] p-8 text-center text-slate-300">
                    Loading supplier ledger...
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="grid grid-cols-2 gap-3">
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
                      <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                      <Input
                        value={supplierLedgerSearch}
                        onChange={(event) => setSupplierLedgerSearch(event.target.value)}
                        placeholder={
                          supplierDetailTab === "ledger"
                            ? "Search ledger date, type, mode, bill, or amount..."
                            : "Search supplier bill number, date, status, or amount..."
                        }
                        className="h-11 rounded-2xl border-white/10 bg-slate-950/60 pl-9 text-white placeholder:text-slate-500"
                      />
                    </div>

                    {supplierDetailTab === "ledger" ? (
                      filteredSupplierLedgerEntries.length === 0 ? (
                        <div className="rounded-2xl border border-dashed border-white/10 bg-white/[0.04] p-8 text-center text-slate-400">
                          No supplier ledger entries found
                        </div>
                      ) : (
                        <div className={`${floatingPanelClass} p-3`}>
                          <div className="max-h-[26rem] overflow-auto rounded-2xl border border-white/10 bg-slate-950/40">
                            <Table>
                              <TableHeader>
                                <TableRow className="border-white/10 bg-white/[0.04]">
                                  <TableHead>Date</TableHead>
                                  <TableHead>Time</TableHead>
                                  <TableHead>Type</TableHead>
                                  <TableHead>Bill</TableHead>
                                  <TableHead className="text-right">Debit</TableHead>
                                  <TableHead className="text-right">Credit</TableHead>
                                  <TableHead className="text-right">Balance</TableHead>
                                  <TableHead>Mode</TableHead>
                                </TableRow>
                              </TableHeader>
                              <TableBody>
                                {filteredSupplierLedgerEntries.map((entry) => {
                                  const createdAt = formatDateTimeValue(entry.created_at || entry.date)

                                  return (
                                    <TableRow key={entry.id} className="border-white/10">
                                      <TableCell className="whitespace-nowrap">{formatDateValue(entry.date)}</TableCell>
                                      <TableCell className="whitespace-nowrap">{createdAt.time}</TableCell>
                                      <TableCell className="font-medium">{getSupplierLedgerTypeLabel(entry.type)}</TableCell>
                                      <TableCell>{entry.bill_number || "-"}</TableCell>
                                      <TableCell className="whitespace-nowrap text-right text-rose-300">
                                        {Number(entry.debit || 0) > 0 ? formatCurrency(entry.debit) : "-"}
                                      </TableCell>
                                      <TableCell className="whitespace-nowrap text-right text-emerald-300">
                                        {Number(entry.credit || 0) > 0 ? formatCurrency(entry.credit) : "-"}
                                      </TableCell>
                                      <TableCell className="whitespace-nowrap text-right text-white">
                                        {formatCurrency(entry.balance)}
                                      </TableCell>
                                      <TableCell className="uppercase text-slate-300">{entry.mode || "-"}</TableCell>
                                    </TableRow>
                                  )
                                })}
                              </TableBody>
                            </Table>
                          </div>
                        </div>
                      )
                    ) : filteredSupplierInvoices.length === 0 ? (
                      <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/40 p-8 text-center text-slate-400">
                        No supplier bills found
                      </div>
                    ) : (
                      <div className="grid gap-3 lg:grid-cols-2">
                        {filteredSupplierInvoices.map((invoice) => (
                          <div
                            key={invoice.id}
                            className="rounded-3xl border border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.12),_transparent_28%),linear-gradient(135deg,#020617_0%,#0f172a_54%,#111827_100%)] p-4 text-white shadow-[0_18px_48px_rgba(15,23,42,0.24)]"
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div>
                                <p className="text-xs uppercase tracking-[0.24em] text-slate-400">Supplier Bill</p>
                                <p className="mt-2 text-base font-semibold">{invoice.bill_number || invoice.id}</p>
                                <p className="mt-1 text-sm text-slate-300">
                                  Issue Date: {formatDateValue(invoice.invoice_date)}
                                </p>
                              </div>
                              <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${getInvoiceStatusClass(invoice.status)}`}>
                                {String(invoice.status || "pending").toUpperCase()}
                              </span>
                            </div>

                            <div className="mt-4 rounded-2xl border border-white/10 bg-white/5 p-4 backdrop-blur">
                              <div className="flex items-start justify-between gap-3 border-b border-white/10 pb-3">
                                <div>
                                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Supplier</p>
                                  <p className="mt-1 font-semibold">{invoice.supplier_name || viewSupplierDetails.name}</p>
                                </div>
                                <div className="text-right">
                                  <p className="text-xs uppercase tracking-[0.2em] text-slate-400">Created</p>
                                  <p className="mt-1 text-sm text-slate-200">{formatDateValue(invoice.created_at)}</p>
                                </div>
                              </div>

                              <div className="mt-3 grid grid-cols-3 gap-2">
                                <BillStat label="Total" value={formatCurrency(invoice.total_amount)} />
                                <BillStat label="Paid" value={formatCurrency(invoice.paid_amount)} tone="success" />
                                <BillStat label="Pending" value={formatCurrency(invoice.pending_amount)} tone="danger" />
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>
    </div>
  )
}

function ActionFeatureCard({
  icon,
  badge,
  title,
  description,
  actionLabel,
  tone = "indigo",
  onClick,
}) {
  const toneClasses = {
    indigo:
      "border-indigo-400/18 bg-[linear-gradient(135deg,rgba(79,70,229,0.12),rgba(255,255,255,0.018))]",
    emerald:
      "border-emerald-400/18 bg-[linear-gradient(135deg,rgba(16,185,129,0.1),rgba(255,255,255,0.018))]",
  }

  return (
    <button
      type="button"
      onClick={onClick}
      className={`min-h-[168px] rounded-[18px] border p-4 text-left text-white shadow-[0_18px_36px_rgba(2,6,23,0.1)] backdrop-blur-xl transition hover:translate-y-[-1px] md:min-h-[128px] md:p-3.5 ${toneClasses[tone] || toneClasses.indigo}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-[14px] bg-white/[0.06] text-white">
          {icon}
        </div>
        <span className="rounded-full border border-white/10 bg-white/10 px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-100">
          {badge}
        </span>
      </div>

      <div className="mt-4 space-y-2 md:mt-3">
        <p className="text-xl font-semibold leading-tight md:text-lg">{title}</p>
        <p className="text-sm leading-7 text-slate-200/85 md:line-clamp-2 md:leading-6">{description}</p>
        <p className="pt-1 text-xs font-semibold uppercase tracking-[0.28em] text-slate-100">
          {actionLabel}
        </p>
      </div>
    </button>
  )
}

function ModalStat({ label, value, tone = "default" }) {
  const toneClasses = {
    default: "bg-white/[0.035] text-white",
    success: "bg-emerald-500/[0.12] text-emerald-300",
    warning: "bg-amber-500/[0.12] text-amber-200",
    danger: "bg-rose-500/[0.12] text-rose-300",
  }

  return (
    <div className={`rounded-[16px] border border-white/8 px-3 py-2 backdrop-blur-xl ${toneClasses[tone] || toneClasses.default}`}>
      <p className="text-[10px] uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-semibold">{value}</p>
    </div>
  )
}

function BillStat({ label, value, tone = "default" }) {
  const toneClasses = {
    default: "bg-white/[0.035] text-white",
    success: "bg-emerald-500/[0.12] text-emerald-300",
    danger: "bg-rose-500/[0.12] text-rose-300",
  }

  return (
    <div className={`rounded-[16px] p-3 ${toneClasses[tone] || toneClasses.default}`}>
      <p className="text-[10px] uppercase tracking-[0.14em] text-slate-400">{label}</p>
      <p className="mt-2 text-sm font-semibold">{value}</p>
    </div>
  )
}
