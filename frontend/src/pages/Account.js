"use client"

import { useCallback, useEffect, useState } from "react"
import axios from "axios"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "sonner"
import {
  Download,
  TrendingUp,
  TrendingDown,
  Wallet,
  Building2,
  Users,
  FileText,
  Search,
  ChevronLeft,
  ChevronRight,
  Receipt
} from "lucide-react"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const formatInputDate = (date) => {
  const year = date.getFullYear()
  const month = String(date.getMonth() + 1).padStart(2, "0")
  const day = String(date.getDate()).padStart(2, "0")
  return `${year}-${month}-${day}`
}

const getCurrentMonthRange = () => {
  const today = new Date()
  return {
    start: formatInputDate(new Date(today.getFullYear(), today.getMonth(), 1)),
    end: formatInputDate(today),
  }
}
 const formatCurrency = (amount) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      minimumFractionDigits: 2
    }).format(amount || 0)
  }
const buildDateQuery = (startDate, endDate) => {
  const params = new URLSearchParams()
  if (startDate) params.append("start_date", startDate)
  if (endDate) params.append("end_date", endDate)
  return params.toString()
}

const matchesSearchText = (fields, rawSearch) => {
  const query = String(rawSearch || "").trim().toLowerCase()
  if (!query) return true

  return fields.some((field) => String(field || "").toLowerCase().includes(query))
}

const getLedgerTypeMeta = (type) => {
  const normalized = String(type || "").toLowerCase()

  switch (normalized) {
    case "sale":
      return { label: "Sale Payment", className: "bg-green-100 text-green-700" }
    case "sale_payment":
      return { label: "Sale Payment", className: "bg-emerald-100 text-emerald-700" }
    case "advance_in":
      return { label: "Advance In", className: "bg-blue-100 text-blue-700" }
    case "advance_used":
      return { label: "Advance Used", className: "bg-amber-100 text-amber-700" }
    case "payment_in":
      return { label: "Receipt", className: "bg-sky-100 text-sky-700" }
    case "payment_out":
      return { label: "Supplier Pay", className: "bg-orange-100 text-orange-700" }
    case "supplier_bill":
      return { label: "Supplier Bill", className: "bg-indigo-100 text-indigo-700" }
    case "expense":
      return { label: "Expense", className: "bg-red-100 text-red-700" }
    case "salary_expense":
      return { label: "Salary", className: "bg-rose-100 text-rose-700" }
    case "supplier_payment":
      return { label: "Supplier Pay", className: "bg-red-100 text-red-700" }
    default:
      return {
        label: String(type || "Other").replaceAll("_", " "),
        className: "bg-gray-100 text-gray-700",
      }
  }
}

const getSupplierBillStatusMeta = (status) => {
  const normalized = String(status || "pending").toLowerCase()

  if (normalized === "paid") {
    return { label: "PAID", className: "bg-emerald-100 text-emerald-700" }
  }

  if (normalized === "partial") {
    return { label: "PARTIAL", className: "bg-amber-100 text-amber-700" }
  }

  return { label: "PENDING", className: "bg-rose-100 text-rose-700" }
}

const getSalaryEntryMeta = (entry) => {
  const paidAmount = Number(entry?.paid_amount || 0)
  const pendingAmount = Number(entry?.pending_amount || 0)

  if (paidAmount > 0 && pendingAmount <= 0) {
    return { label: "PAID", className: "bg-emerald-100 text-emerald-700" }
  }

  if (paidAmount > 0) {
    return { label: "PARTIAL", className: "bg-amber-100 text-amber-700" }
  }

  return { label: "OPEN", className: "bg-slate-200 text-slate-700" }
}

const buildStatsCards = (data, overallData) => {
  if (!data) return []

  // 🔁 FILTERED VALUES
  const totalSales = Number(data.total_sales || 0)
  const cashBalance = Number(data.cash_balance || 0)
  const bankBalance = Number(data.bank_balance || 0)
  const income = Number(data.income || 0)
  const expense = Number(data.total_expenses ?? data.expense ?? 0)
  const salary = Number(data.salary_expenses || 0)

  // 🔥 ALL TIME VALUES
  const totalSalesAll = Number(overallData?.total_sales || 0)
  const cashAll = Number(overallData?.cash_balance || 0)
  const bankAll = Number(overallData?.bank_balance || 0)
  const incomeAll = Number(overallData?.income || 0)
  const expenseAll = Number(overallData?.total_expenses ?? overallData?.expense ?? 0)
  const salaryAll = Number(overallData?.salary_expenses || 0)

  return [
    {
      key: "total-sales",
      label: "Total Sales",
      hint: `All Time: ${formatCurrency(totalSalesAll)}`, // ✅ ALWAYS ALL TIME
      value: totalSales,
      icon: FileText,
      borderClass: "border-l-indigo-500",
      textClass: "text-indigo-600",
      iconClass: "text-indigo-600",
    },
    {
      key: "total-received",
      label: "Total Amount Received",
      hint: `All Time: ${formatCurrency(incomeAll)}`,
      value: income,
      icon: TrendingUp,
      borderClass: "border-l-purple-500",
      textClass: "text-purple-600",
      iconClass: "text-purple-600",
    },
    {
      key: "cash-balance",
      label: "Cash Balance",
      hint: `All Time: ${formatCurrency(cashAll)}`,
      value: cashBalance,
      icon: Wallet,
      borderClass: "border-l-green-500",
      textClass: "text-green-600",
      iconClass: "text-green-600",
    },
    {
      key: "bank-balance",
      label: "Bank Balance",
      hint: `All Time: ${formatCurrency(bankAll)}`,
      value: bankBalance,
      icon: Building2,
      borderClass: "border-l-blue-500",
      textClass: "text-blue-600",
      iconClass: "text-blue-600",
    },
    {
      key: "total-expense",
      label: "Total Expenses",
      hint: `All Time: ${formatCurrency(expenseAll)}`,
      value: expense,
      icon: Receipt,
      borderClass: "border-l-orange-500",
      textClass: "text-orange-600",
      iconClass: "text-orange-600",
    },
    {
      key: "salary-expense",
      label: "Salary",
      hint: `All Time: ${formatCurrency(salaryAll)}`,
      value: salary,
      icon: Users,
      borderClass: "border-l-rose-500",
      textClass: "text-rose-600",
      iconClass: "text-rose-600",
    },
  ]
}
export default function Accounts() {
  const [ledger, setLedger] = useState({ data: [], pagination: {} })
  const [cashbook, setCashbook] = useState({ data: [], pagination: {} })
  const [bankbook, setBankbook] = useState({ data: [], pagination: {} })
  const [gstLedger, setGstLedger] = useState({ data: [], pagination: {} })
  const [expenses, setExpenses] = useState({ data: [], pagination: {} })
  const [supplierLedger, setSupplierLedger] = useState({ data: [], pagination: {}, summary: {} })
  const [summary, setSummary] = useState(null)
  const [suppliers, setSuppliers] = useState([])
  const [employees, setEmployees] = useState([])
  const [salaryPayments, setSalaryPayments] = useState([])
  const [powerBills, setPowerBills] = useState([])
  const [supplierInvoices, setSupplierInvoices] = useState([])
  const [expenseCategories, setExpenseCategories] = useState([])
  const [cheques, setCheques] = useState({ data: [], pagination: {} })

  const [tab, setTab] = useState("ledger")
  const [page, setPage] = useState(1)
  const [startDate, setStartDate] = useState(() => getCurrentMonthRange().start)
  const [endDate, setEndDate] = useState(() => getCurrentMonthRange().end)
  const [paymentModeFilter, setPaymentModeFilter] = useState("all")
  const [supplierBillLookup, setSupplierBillLookup] = useState("")
  const [supplierPaymentLookup, setSupplierPaymentLookup] = useState("")
  const [salaryDialogLookup, setSalaryDialogLookup] = useState("")
  const [salaryHistoryLookup, setSalaryHistoryLookup] = useState("")
  const [filtersOpen, setFiltersOpen] = useState(true)

  // Form States
  const [expenseTitle, setExpenseTitle] = useState("")
  const [expenseAmount, setExpenseAmount] = useState("")
  const [expenseCategory, setExpenseCategory] = useState("")
  const [expensePaymentMode, setExpensePaymentMode] = useState("cash")
  const [expenseDescription, setExpenseDescription] = useState("")
  const [expenseDate, setExpenseDate] = useState(() => formatInputDate(new Date()))

  const [supplierPaymentSupplier, setSupplierPaymentSupplier] = useState("")
  const [supplierPaymentAmount, setSupplierPaymentAmount] = useState("")
  const [supplierPaymentMode, setSupplierPaymentMode] = useState("cash")
  const [supplierPaymentDescription, setSupplierPaymentDescription] = useState("")
  const [supplierPaymentDate, setSupplierPaymentDate] = useState(() => formatInputDate(new Date()))

  const [employeeName, setEmployeeName] = useState("")
  const [employeeRole, setEmployeeRole] = useState("")
  const [employeeSalary, setEmployeeSalary] = useState("")
  const [salaryEmployeeId, setSalaryEmployeeId] = useState("")
  const [salaryAmount, setSalaryAmount] = useState("")
  const [salaryPaymentMode, setSalaryPaymentMode] = useState("cash")
  const [salaryNotes, setSalaryNotes] = useState("")
  const [salaryDate, setSalaryDate] = useState(() => formatInputDate(new Date()))

  const [powerBillAmount, setPowerBillAmount] = useState("")
  const [powerBillMode, setPowerBillMode] = useState("cash")
  const [powerBillDescription, setPowerBillDescription] = useState("")
  const [powerBillDate, setPowerBillDate] = useState(() => formatInputDate(new Date()))

  const [supplierBillSupplierId, setSupplierBillSupplierId] = useState("")
  const [supplierBillAmount, setSupplierBillAmount] = useState("")
  const [supplierBillDate, setSupplierBillDate] = useState(() => formatInputDate(new Date()))
  const [isAddExpenseOpen, setIsAddExpenseOpen] = useState(false)
  const [isPaySupplierOpen, setIsPaySupplierOpen] = useState(false)
  const [isAddEmployeeOpen, setIsAddEmployeeOpen] = useState(false)
  const [isPaySalaryOpen, setIsPaySalaryOpen] = useState(false)
  const [isAddPowerBillOpen, setIsAddPowerBillOpen] = useState(false)
  const [isAddSupplierBillOpen, setIsAddSupplierBillOpen] = useState(false)

  const token = typeof window !== "undefined" ? localStorage.getItem("token") : ""

  const fetchAllData = useCallback(async () => {
    try {
      const headers = { Authorization: `Bearer ${token}` }
      const filteredStatsQuery = buildDateQuery(startDate, endDate)
      const filteredStatsSuffix = filteredStatsQuery ? `?${filteredStatsQuery}` : ""

      // Fetch supporting data for account actions
      const [
        suppliersRes,
        categoriesRes,
        employeesRes,
        salaryPaymentsRes,
        powerBillsRes,
        supplierInvoicesRes,
      ] = await Promise.all([
        axios.get(`${API}/suppliers`, { headers }),
        axios.get(`${API}/expense-categories`, { headers }),
        axios.get(`${API}/employees`, { headers }),
        axios.get(`${API}/salary-payments`, { headers }),
        axios.get(`${API}/power-bills`, { headers }),
        axios.get(`${API}/supplier-invoices`, { headers }),
      ])

      setSuppliers(suppliersRes.data)
      setExpenseCategories(categoriesRes.data)
      setEmployees(employeesRes.data || [])
      setSalaryPayments(salaryPaymentsRes.data || [])
      setPowerBills(powerBillsRes.data || [])
      setSupplierInvoices(supplierInvoicesRes.data || [])

      // Build query params
      const params = new URLSearchParams()
      params.append('page', page)
      params.append('limit', 20)
      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)
      if (paymentModeFilter !== 'all') params.append('payment_mode', paymentModeFilter)

      // Fetch data based on tab
      if (tab === "ledger") {
        const res = await axios.get(`${API}/accounts/ledger?${params}`, { headers })
        setLedger(res.data)
      } else if (tab === "cashbook") {
        const res = await axios.get(`${API}/accounts/cashbook?${params}`, { headers })
        setCashbook(res.data)
      } else if (tab === "bankbook") {
        const res = await axios.get(`${API}/accounts/bankbook?${params}`, { headers })
        setBankbook(res.data)
      } else if (tab === "gst") {
        const res = await axios.get(`${API}/accounts/gst-ledger?${params}`, { headers })
        setGstLedger(res.data)
      } else if (tab === "expenses") {
        const res = await axios.get(`${API}/expenses?${params}`, { headers })
        setExpenses(res.data)
      } else if (tab === "suppliers") {
        const res = await axios.get(`${API}/accounts/supplier-ledger?${params}`, { headers })
        setSupplierLedger(res.data)
      } else if (tab === "cheques") {
        const res = await axios.get(`${API}/cheques?${params}`, { headers })
        setCheques(res.data)
      }

      // Always fetch summary and profit/loss
      const [summaryRes] = await Promise.all([
        axios.get(`${API}/accounts/summary${filteredStatsSuffix}`, { headers }),
      ])

      setSummary(summaryRes.data)

    } catch (error) {
      toast.error("Failed to load accounts data")
      console.error(error)
    }
  }, [endDate, page, paymentModeFilter, startDate, tab, token])

  useEffect(() => {
    fetchAllData()
  }, [fetchAllData])

  // Add Expense
  const addExpense = async () => {
    if (!expenseTitle || !expenseAmount) {
      toast.error("Enter expense details")
      return
    }

    try {
      await axios.post(
        `${API}/expenses`,
        {
          title: expenseTitle,
          amount: Number(expenseAmount),
          category_id: expenseCategory || null,
          payment_mode: expensePaymentMode,
          description: expenseDescription,
          expense_date: expenseDate
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Expense added successfully")
      setExpenseTitle("")
      setExpenseAmount("")
      setExpenseCategory("")
      setExpenseDescription("")
      setExpenseDate(formatInputDate(new Date()))
      setIsAddExpenseOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error("Failed to add expense")
      console.error(error)
    }
  }

  // Pay Supplier
  const paySupplier = async () => {
    const selectedInvoice = supplierInvoices.find((invoice) => invoice.id === supplierPaymentSupplier)
    const pendingAmount = Number(selectedInvoice?.pending_amount || 0)

    if (!supplierPaymentSupplier || !supplierPaymentAmount) {
      toast.error("Select a pending bill and enter amount")
      return
    }

    if (!selectedInvoice) {
      toast.error("Pending supplier bill not found")
      return
    }

    if (pendingAmount <= 0) {
      toast.error("No pending balance available for this supplier bill")
      return
    }

    if (Number(supplierPaymentAmount) > pendingAmount) {
      toast.error(`Amount cannot exceed pending balance of ${formatCurrency(pendingAmount)}`)
      return
    }

    try {
      await axios.post(
        `${API}/supplier-invoices/${supplierPaymentSupplier}/pay`,
        {
        amount: Number(supplierPaymentAmount),
        payment_mode: supplierPaymentMode,
          notes: supplierPaymentDescription || null,
        payment_date: supplierPaymentDate
        },
        {
        headers: { Authorization: `Bearer ${token}` }
        }
      )

      toast.success("Supplier bill payment recorded")
      setSupplierPaymentSupplier("")
      setSupplierPaymentAmount("")
      setSupplierPaymentMode("cash")
      setSupplierPaymentDescription("")
      setSupplierPaymentLookup("")
      setSupplierPaymentDate(formatInputDate(new Date()))
      setIsPaySupplierOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to pay supplier")
      console.error(error)
    }
  }

  const addEmployee = async () => {
    if (!employeeName || !employeeSalary) {
      toast.error("Enter employee name and salary")
      return
    }

    try {
      await axios.post(
        `${API}/employees`,
        {
          name: employeeName,
          role: employeeRole || null,
          salary: Number(employeeSalary),
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Employee added successfully")
      setEmployeeName("")
      setEmployeeRole("")
      setEmployeeSalary("")
      setIsAddEmployeeOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add employee")
    }
  }

  const paySalary = async () => {
    if (!salaryEmployeeId || !salaryAmount) {
      toast.error("Select employee and enter amount")
      return
    }

    const pendingAmount = Number(selectedSalaryEmployee?.pending_amount ?? 0)

    if (!selectedSalaryEmployee) {
      toast.error("Pending salary employee not found")
      return
    }

    if (pendingAmount <= 0) {
      toast.error("Salary already fully paid for this month")
      return
    }

    if (Number(salaryAmount) > pendingAmount) {
      toast.error(`Amount cannot exceed pending salary of ${formatCurrency(pendingAmount)}`)
      return
    }

    try {
      await axios.post(
        `${API}/employees/${salaryEmployeeId}/salary-payments/pay`,
        {
          amount: Number(salaryAmount),
          payment_mode: salaryPaymentMode,
          payment_date: salaryDate,
          notes: salaryNotes || null,
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Salary payment recorded")
      setSalaryEmployeeId("")
      setSalaryAmount("")
      setSalaryPaymentMode("cash")
      setSalaryNotes("")
      setSalaryDate(formatInputDate(new Date()))
      setSalaryDialogLookup("")
      setIsPaySalaryOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to pay salary")
    }
  }

  const addPowerBill = async () => {
    if (!powerBillAmount) {
      toast.error("Enter power bill amount")
      return
    }

    try {
      await axios.post(
        `${API}/power-bills`,
        {
          amount: Number(powerBillAmount),
          payment_mode: powerBillMode,
          description: powerBillDescription || null,
          expense_date: powerBillDate,
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Power bill added")
      setPowerBillAmount("")
      setPowerBillMode("cash")
      setPowerBillDescription("")
      setPowerBillDate(formatInputDate(new Date()))
      setIsAddPowerBillOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add power bill")
    }
  }

  const addSupplierBill = async () => {
    if (!supplierBillSupplierId || !supplierBillAmount) {
      toast.error("Select supplier and enter bill amount")
      return
    }

    try {
      await axios.post(
        `${API}/supplier-invoices`,
        {
          supplier_id: supplierBillSupplierId,
          total_amount: Number(supplierBillAmount),
          invoice_date: supplierBillDate,
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Supplier bill added")
      setSupplierBillSupplierId("")
      setSupplierBillAmount("")
      setSupplierBillDate(formatInputDate(new Date()))
      setSupplierBillLookup("")
      setIsAddSupplierBillOpen(false)
      fetchAllData()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to add supplier bill")
    }
  }

  // Export Functions
  const exportLedger = async () => {
    try {
      const params = new URLSearchParams()
      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)

      const response = await axios.get(`${API}/accounts/export/ledger?${params}`, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob'
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `ledger_${new Date().toISOString()}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()

      toast.success("Ledger exported successfully")
    } catch (error) {
      toast.error("Failed to export ledger")
    }
  }

  const exportExpenses = async () => {
    try {
      const params = new URLSearchParams()
      if (startDate) params.append('start_date', startDate)
      if (endDate) params.append('end_date', endDate)

      const response = await axios.get(`${API}/accounts/export/expenses?${params}`, {
        headers: { Authorization: `Bearer ${token}` },
        responseType: 'blob'
      })

      const url = window.URL.createObjectURL(new Blob([response.data]))
      const link = document.createElement('a')
      link.href = url
      link.setAttribute('download', `expenses_${new Date().toISOString()}.csv`)
      document.body.appendChild(link)
      link.click()
      link.remove()

      toast.success("Expenses exported successfully")
    } catch (error) {
      toast.error("Failed to export expenses")
    }
  }

  // Quick Date Filters
  const setQuickDateFilter = (filter) => {
    const today = new Date()
    let start, end

    switch (filter) {
      case 'today':
        start = end = formatInputDate(today)
        break
      case 'week':
        start = formatInputDate(new Date(today.getFullYear(), today.getMonth(), today.getDate() - 7))
        end = formatInputDate(new Date())
        break
      case 'month':
        start = formatInputDate(new Date(today.getFullYear(), today.getMonth(), 1))
        end = formatInputDate(new Date())
        break
      case 'all':
        start = end = ""
        break
    }

    setStartDate(start)
    setEndDate(end)
    setPage(1)
  }

 

  const formatDate = (dateString) => {
    if (!dateString) return '-'
    return new Date(dateString).toLocaleDateString('en-IN', {
      day: '2-digit',
      month: 'short',
      year: 'numeric'
    })
  }
  const formatTime = (dateString) => {
    if (!dateString) return "-"
    return new Date(dateString).toLocaleTimeString("en-IN", {
      hour: "2-digit",
      minute: "2-digit",
    })
  }

  const getSupplierLookupLabel = (supplier) =>
    [supplier?.name, supplier?.phone, supplier?.email].filter(Boolean).join(" | ")

  const getSupplierInvoiceLookupLabel = (invoice) =>
    [
      invoice?.bill_number || invoice?.id,
      invoice?.supplier_name,
      `Pending ${formatCurrency(invoice?.pending_amount || 0)}`,
    ]
      .filter(Boolean)
      .join(" | ")

  const getCurrentData = () => {
    switch (tab) {
      case 'ledger': return ledger
      case 'cashbook': return cashbook
      case 'bankbook': return bankbook
      case 'gst': return gstLedger
      case 'expenses': return expenses
      case 'suppliers': return supplierLedger
      case 'cheques': return cheques
      case 'salary':
      case 'power':
        return { data: [], pagination: {} }
      default: return { data: [], pagination: {} }
    }
  }

  const currentData = getCurrentData()
  const currentMonthRange = getCurrentMonthRange()
  const isCurrentMonthFilter =
    startDate === currentMonthRange.start && endDate === currentMonthRange.end
  const filteredStats = buildStatsCards(summary?.filtered, summary?.overall)
  const featuredProfit = summary?.filtered
    ? {
        value: Number(summary.filtered?.profit || 0),
        allTimeValue: Number(summary.overall?.profit || 0),
      }
    : null
  const salaryEntries = [...salaryPayments]
    .filter((entry) =>
      matchesSearchText(
        [
          entry?.payment_date,
          entry?.employee_name,
          entry?.employee_role,
          entry?.payment_mode,
          entry?.notes,
          entry?.salary_period,
          entry?.total_salary,
          entry?.paid_amount,
          entry?.pending_amount,
        ],
        salaryHistoryLookup
      )
    )
    .sort((leftEntry, rightEntry) => {
      const dateCompare = String(rightEntry?.payment_date || "").localeCompare(String(leftEntry?.payment_date || ""))
      if (dateCompare !== 0) return dateCompare

      return String(rightEntry?.created_at || "").localeCompare(String(leftEntry?.created_at || ""))
    })
  const salaryEmployeeOptions = [...employees]
    .filter((employee) =>
      matchesSearchText(
        [
          employee?.name,
          employee?.role,
          employee?.salary_period,
          employee?.pending_amount,
        ],
        salaryDialogLookup
      )
    )
    .filter((employee) => Number(employee?.pending_amount ?? employee?.salary ?? 0) > 0)
    .sort((leftEmployee, rightEmployee) =>
      String(leftEmployee?.name || "").localeCompare(String(rightEmployee?.name || ""), "en", {
        sensitivity: "base",
      })
    )
  const selectedSalaryEmployee = employees.find((employee) => employee.id === salaryEmployeeId) || null
  const supplierSelectOptions = [...suppliers]
    .filter((supplier) =>
      matchesSearchText(
        [supplier?.name, supplier?.phone, supplier?.email, supplier?.address, getSupplierLookupLabel(supplier)],
        supplierBillLookup
      )
    )
    .sort((leftSupplier, rightSupplier) =>
      String(leftSupplier?.name || "").localeCompare(String(rightSupplier?.name || ""), "en", {
        sensitivity: "base",
      })
    )
  const sortedSupplierInvoices = [...supplierInvoices]
    .sort((leftInvoice, rightInvoice) => {
      const dateCompare = String(rightInvoice?.invoice_date || "").localeCompare(String(leftInvoice?.invoice_date || ""))
      if (dateCompare !== 0) return dateCompare

      return String(leftInvoice?.supplier_name || "").localeCompare(String(rightInvoice?.supplier_name || ""), "en", {
        sensitivity: "base",
      })
    })
  const pendingSupplierInvoices = sortedSupplierInvoices.filter((invoice) => Number(invoice.pending_amount || 0) > 0)
  const pendingSupplierPaymentOptions = pendingSupplierInvoices.filter((invoice) =>
    matchesSearchText(
      [
        invoice?.bill_number,
        invoice?.supplier_name,
        invoice?.invoice_date,
        invoice?.status,
        invoice?.pending_amount,
        getSupplierInvoiceLookupLabel(invoice),
      ],
      supplierPaymentLookup
    )
  )
  const selectedSupplierPaymentInvoice = pendingSupplierInvoices.find(
    (invoice) => invoice.id === supplierPaymentSupplier
  )
  const supplierLedgerEntries = supplierLedger?.data || []

  return (
    <div className="p-2 sm:p-4 md:p-6 space-y-3 sm:space-y-6 bg-background min-h-screen" data-testid="accounts-page">

      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 sm:gap-4">
        <h1 className="text-lg sm:text-2xl md:text-3xl font-bold text-white-900">Accounts & Ledger</h1>
        <div className="flex flex-col sm:flex-row gap-2 w-full sm:w-auto">
          <Button
            variant="outline"
            onClick={exportLedger}
            className="gap-2 w-full sm:w-auto text-sm"
            data-testid="export-ledger-btn"
          >
            <Download className="w-4 h-4" />
            <span className="hidden sm:inline">Export Ledger</span>
            <span className="sm:hidden">Export</span>
          </Button>
          {tab === 'expenses' && (
            <Button
              variant="outline"
              onClick={exportExpenses}
              className="gap-2 w-full sm:w-auto text-sm"
              data-testid="export-expenses-btn"
            >
              <Download className="w-4 h-4" />
              <span className="hidden sm:inline">Export Expenses</span>
              <span className="sm:hidden">Export</span>
            </Button>
          )}
        </div>
      </div>

      {/* Stats */}
      {(featuredProfit || filteredStats.length > 0) && (
        <div className="space-y-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <h2 className="text-base md:text-lg font-semibold">
                {isCurrentMonthFilter ? "Current Month Stats" : "Selected Period Stats"}
              </h2>
              <p className="text-xs md:text-sm text-gray-500">
                {startDate && endDate ? `${formatDate(startDate)} to ${formatDate(endDate)}` : "All-time totals"}
              </p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 lg:grid-cols-3">
            {filteredStats.map((stat) => {
              const Icon = stat.icon

              return (
                <Card key={stat.key} className={`border-l-4 ${stat.borderClass}`}>
                  <CardHeader className="pb-1 pt-2 px-3 sm:pb-2 sm:pt-4 sm:px-4">
                    <div className="flex items-center justify-between gap-2">
                      <CardTitle className="truncate text-xs font-medium text-gray-600 md:text-sm">
                        {stat.label}
                      </CardTitle>
                      <Icon className={`h-4 w-4 flex-shrink-0 md:h-5 md:w-5 ${stat.iconClass}`} />
                    </div>
                    <p className="mt-1 text-[10px] text-gray-500 sm:text-xs">{stat.hint}</p>
                  </CardHeader>
                  <CardContent className="px-3 py-2 sm:px-4">
                    <p className={`break-words text-sm font-bold sm:text-base md:text-lg ${stat.textClass}`} data-testid={stat.key}>
                      {formatCurrency(stat.value)}
                    </p>
                  </CardContent>
                </Card>
              )
            })}
          </div>

          {featuredProfit && (
            <Card className="overflow-hidden border border-emerald-500/60 bg-[radial-gradient(circle_at_top_right,_rgba(34,197,94,0.16),_transparent_34%),linear-gradient(135deg,_rgba(6,78,59,0.96),_rgba(4,47,46,0.96))] shadow-[0_14px_42px_rgba(16,185,129,0.14)]">
              <CardContent className="flex flex-col gap-4 px-5 py-3.5 text-white sm:flex-row sm:items-center sm:justify-between md:px-6">
                <div className="space-y-1.5">
                  <p className="text-sm tracking-[0.16em] text-emerald-100/80">Net Profit / Loss</p>
                  <p
                    className={`text-2xl font-bold tracking-tight sm:text-3xl ${
                      featuredProfit.value >= 0 ? "text-emerald-300" : "text-rose-300"
                    }`}
                  >
                    {formatCurrency(featuredProfit.value)}
                  </p>
                  <p className="text-xs text-emerald-50/75">
                    Formula: cash balance + bank balance - expenses - supplier pay - salary
                  </p>
                </div>

                <div className="flex h-12 w-12 items-center justify-center rounded-full bg-emerald-400/15 ring-1 ring-inset ring-emerald-200/20 sm:h-14 sm:w-14">
                  {featuredProfit.value >= 0 ? (
                    <TrendingUp className="h-6 w-6 text-emerald-300 sm:h-7 sm:w-7" />
                  ) : (
                    <TrendingDown className="h-6 w-6 text-rose-300 sm:h-7 sm:w-7" />
                  )}
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Quick Actions */}
      <div className="grid grid-cols-2 gap-3 md:gap-4">
        <Dialog open={isAddExpenseOpen} onOpenChange={setIsAddExpenseOpen}>
          <DialogTrigger asChild>
            <Card className="col-span-2 min-h-[128px] cursor-pointer border-2 border-dashed border-gray-300 transition-shadow hover:border-orange-500 hover:shadow-lg">
              <CardContent className="flex h-full items-center p-5 md:p-6">
                <div className="flex items-center gap-3 md:gap-4">
                  <div className="p-2 md:p-3 bg-orange-100 rounded-full flex-shrink-0">
                    <Receipt className="w-5 md:w-6 h-5 md:h-6 text-orange-600" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-semibold text-base md:text-lg">Add Expense</h3>
                    <p className="text-xs md:text-sm text-gray-600 truncate">Record new</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </DialogTrigger>
          <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Add New Expense</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <div>
                <Label>Title *</Label>
                <Input
                  placeholder="Expense title"
                  value={expenseTitle}
                  onChange={(e) => setExpenseTitle(e.target.value)}
                  data-testid="expense-title-input"
                />
              </div>
              <div>
                <Label>Amount *</Label>
                <Input
                  type="number"
                  placeholder="0.00"
                  value={expenseAmount}
                  onChange={(e) => setExpenseAmount(e.target.value)}
                  data-testid="expense-amount-input"
                />
              </div>
              <div>
                <Label>Category</Label>
                <Select value={expenseCategory} onValueChange={setExpenseCategory}>
                  <SelectTrigger data-testid="expense-category-select">
                    <SelectValue placeholder="Select category" />
                  </SelectTrigger>
                  <SelectContent>
                    {expenseCategories.map(cat => (
                      <SelectItem key={cat.id} value={cat.id}>{cat.name}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Payment Mode</Label>
                <Select value={expensePaymentMode} onValueChange={setExpensePaymentMode}>
                  <SelectTrigger>
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
              <div>
                <Label>Date</Label>
                <Input
                  type="date"
                  value={expenseDate}
                  onChange={(e) => setExpenseDate(e.target.value)}
                />
              </div>
              <div>
                <Label>Description</Label>
                <Input
                  placeholder="Optional description"
                  value={expenseDescription}
                  onChange={(e) => setExpenseDescription(e.target.value)}
                />
              </div>
              <Button onClick={addExpense} className="w-full" data-testid="add-expense-submit-btn">
                Add Expense
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={isAddSupplierBillOpen} onOpenChange={setIsAddSupplierBillOpen}>
          <DialogTrigger asChild>
            <Card className="cursor-pointer border-2 border-dashed border-gray-300 transition-shadow hover:border-indigo-500 hover:shadow-lg">
              <CardContent className="p-4">
                <div className="flex items-center gap-3 md:gap-4">
                  <div className="rounded-full bg-indigo-100 p-2 md:p-3">
                    <FileText className="h-5 w-5 text-indigo-600 md:h-6 md:w-6" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="text-base font-semibold md:text-lg">Add Supplier Bill</h3>
                    <p className="truncate text-xs text-gray-600 md:text-sm">Create pending supplier bill</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </DialogTrigger>
          <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Add Supplier Bill</DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <div>
                <Label>Supplier</Label>
                <div className="relative mt-2">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    list="account-supplier-bill-options"
                    value={supplierBillLookup}
                    onChange={(e) => {
                      const value = e.target.value
                      setSupplierBillLookup(value)
                      const matchedSupplier = suppliers.find(
                        (supplier) => getSupplierLookupLabel(supplier) === value
                      )
                      setSupplierBillSupplierId(matchedSupplier?.id || "")
                    }}
                    placeholder="Search and select supplier"
                    className="pl-9"
                  />
                  <datalist id="account-supplier-bill-options">
                    {supplierSelectOptions.map((supplier) => (
                      <option key={supplier.id} value={getSupplierLookupLabel(supplier)} />
                    ))}
                  </datalist>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  One searchable dropdown for supplier name, phone, or email.
                </p>
              </div>
              <div>
                <Label>Bill Amount</Label>
                <Input type="number" value={supplierBillAmount} onChange={(e) => setSupplierBillAmount(e.target.value)} />
              </div>
              <div>
                <Label>Invoice Date</Label>
                <Input type="date" value={supplierBillDate} onChange={(e) => setSupplierBillDate(e.target.value)} />
              </div>
              <Button onClick={addSupplierBill} className="w-full">Save Supplier Bill</Button>
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={isPaySupplierOpen} onOpenChange={setIsPaySupplierOpen}>
          <DialogTrigger asChild>
            <Card className="cursor-pointer border-2 border-dashed border-gray-300 transition-shadow hover:border-red-500 hover:shadow-lg">
              <CardContent className="p-4">
                <div className="flex items-center gap-3 md:gap-4">
                  <div className="p-2 md:p-3 bg-red-100 rounded-full flex-shrink-0">
                    <Users className="w-5 md:w-6 h-5 md:h-6 text-red-600" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-semibold text-base md:text-lg">Pay Supplier</h3>
                    <p className="text-xs md:text-sm text-gray-600 truncate">Pending bills only</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </DialogTrigger>
          <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle className="text-lg md:text-xl">Pay Supplier</DialogTitle>
            </DialogHeader>
            <div className="space-y-3 md:space-y-4">
              <div>
                <Label>Pending Bill *</Label>
                <div className="relative mt-2">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    list="account-supplier-payment-options"
                    value={supplierPaymentLookup}
                    onChange={(e) => {
                      const value = e.target.value
                      setSupplierPaymentLookup(value)
                      const matchedInvoice = pendingSupplierInvoices.find(
                        (invoice) => getSupplierInvoiceLookupLabel(invoice) === value
                      )
                      setSupplierPaymentSupplier(matchedInvoice?.id || "")
                      setSupplierPaymentAmount(matchedInvoice ? String(matchedInvoice.pending_amount || "") : "")
                    }}
                    placeholder="Search and select pending bill"
                    className="pl-9"
                  />
                  <datalist id="account-supplier-payment-options">
                    {pendingSupplierPaymentOptions.map((invoice) => (
                      <option key={invoice.id} value={getSupplierInvoiceLookupLabel(invoice)} />
                    ))}
                  </datalist>
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Only supplier bills with pending balance can be paid.
                </p>
              </div>
              {selectedSupplierPaymentInvoice ? (
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Pending Balance</p>
                      <p className="text-sm text-emerald-800">{selectedSupplierPaymentInvoice.supplier_name}</p>
                    </div>
                    <p className="text-lg font-bold text-emerald-700">
                      {formatCurrency(selectedSupplierPaymentInvoice.pending_amount)}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-3 text-sm text-slate-600">
                  {pendingSupplierInvoices.length === 0
                    ? "No pending balance available, so payment is disabled."
                    : "Pick a pending bill and the payable amount will auto-fill."}
                </div>
              )}
              <div>
                <Label>Amount *</Label>
                <Input
                  type="number"
                  placeholder="0.00"
                  value={supplierPaymentAmount}
                  onChange={(e) => setSupplierPaymentAmount(e.target.value)}
                  data-testid="supplier-payment-amount-input"
                />
              </div>
              <div>
                <Label>Payment Mode</Label>
                <Select value={supplierPaymentMode} onValueChange={setSupplierPaymentMode}>
                  <SelectTrigger>
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
              <div>
                <Label>Date</Label>
                <Input
                  type="date"
                  value={supplierPaymentDate}
                  onChange={(e) => setSupplierPaymentDate(e.target.value)}
                />
              </div>
              <div>
                <Label>Notes</Label>
                <Input
                  placeholder="Optional notes"
                  value={supplierPaymentDescription}
                  onChange={(e) => setSupplierPaymentDescription(e.target.value)}
                />
              </div>
              <Button
                onClick={paySupplier}
                variant="destructive"
                className="w-full"
                data-testid="pay-supplier-submit-btn"
                disabled={pendingSupplierInvoices.length === 0 || !selectedSupplierPaymentInvoice}
              >
                Pay Supplier
              </Button>
            </div>
          </DialogContent>
        </Dialog>

        <Card
          role="button"
          tabIndex={0}
          onClick={() => setIsAddEmployeeOpen(true)}
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault()
              setIsAddEmployeeOpen(true)
            }
          }}
          className="cursor-pointer border-2 border-dashed border-gray-300 transition-shadow hover:border-sky-500 hover:shadow-lg"
        >
          <CardContent className="p-4">
            <div className="flex items-center gap-3 md:gap-4">
              <div className="rounded-full bg-sky-100 p-2 md:p-3">
                <Users className="h-5 w-5 text-sky-600 md:h-6 md:w-6" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-base font-semibold md:text-lg">Add Employee</h3>
                <p className="truncate text-xs text-gray-600 md:text-sm">Create salary profile</p>
              </div>
            </div>
          </CardContent>
        </Card>

        <Dialog open={isPaySalaryOpen} onOpenChange={setIsPaySalaryOpen}>
          <DialogTrigger asChild>
            <Card className="cursor-pointer border-2 border-dashed border-gray-300 transition-shadow hover:border-rose-500 hover:shadow-lg">
              <CardContent className="p-4">
                <div className="flex items-center gap-3 md:gap-4">
                  <div className="p-2 md:p-3 bg-rose-100 rounded-full flex-shrink-0">
                    <Wallet className="w-5 md:w-6 h-5 md:h-6 text-rose-600" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <h3 className="font-semibold text-base md:text-lg">Pay Salary</h3>
                    <p className="text-xs md:text-sm text-gray-600 truncate">Employee payment</p>
                  </div>
                </div>
              </CardContent>
            </Card>
          </DialogTrigger>
          <DialogContent className="max-w-md max-h-[90vh] overflow-y-auto">
            <DialogHeader>
              <DialogTitle>Pay Salary</DialogTitle>
            </DialogHeader>
            <div className="space-y-3">
              <div>
                <Label>Search Employee</Label>
                <div className="relative mt-2">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={salaryDialogLookup}
                    onChange={(e) => setSalaryDialogLookup(e.target.value)}
                    placeholder="Search employee name, role, or month"
                    className="pl-9"
                  />
                </div>
              </div>
              <div>
                <Label>Employee</Label>
                <Select
                  value={salaryEmployeeId}
                  onValueChange={(value) => {
                    setSalaryEmployeeId(value)
                    const employee = salaryEmployeeOptions.find((option) => option.id === value)
                    setSalaryAmount(employee ? String(Number(employee.pending_amount ?? employee.salary ?? 0)) : "")
                  }}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select pending employee" />
                  </SelectTrigger>
                  <SelectContent>
                    {salaryEmployeeOptions.length === 0 ? (
                      <SelectItem value="no-pending-salary" disabled>
                        No pending salary for this month
                      </SelectItem>
                    ) : (
                      salaryEmployeeOptions.map((employee) => (
                        <SelectItem key={employee.id} value={employee.id}>
                          {employee.name} - Pending {formatCurrency(employee.pending_amount ?? employee.salary ?? 0)}
                        </SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>
              {selectedSalaryEmployee ? (
                <div className="rounded-2xl border border-emerald-200 bg-emerald-50 p-3">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold uppercase tracking-[0.18em] text-emerald-700">Pending Salary</p>
                      <p className="text-sm text-emerald-800">
                        {selectedSalaryEmployee.name} • {selectedSalaryEmployee.salary_period || "Current Month"}
                      </p>
                    </div>
                    <p className="text-lg font-bold text-emerald-700">
                      {formatCurrency(selectedSalaryEmployee.pending_amount ?? selectedSalaryEmployee.salary ?? 0)}
                    </p>
                  </div>
                </div>
              ) : (
                <div className="rounded-2xl border border-dashed border-slate-300 bg-slate-50 p-3 text-sm text-slate-600">
                  {salaryEmployeeOptions.length === 0
                    ? "All salary entries are already paid for this month."
                    : "Select an employee with pending salary to auto-fill the payable amount."}
                </div>
              )}
              <div>
                <Label>Amount</Label>
                <Input type="number" value={salaryAmount} onChange={(e) => setSalaryAmount(e.target.value)} />
              </div>
              <div>
                <Label>Payment Mode</Label>
                <Select value={salaryPaymentMode} onValueChange={setSalaryPaymentMode}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    <SelectItem value="cash">Cash</SelectItem>
                    <SelectItem value="bank">Bank</SelectItem>
                    <SelectItem value="upi">UPI</SelectItem>
                    <SelectItem value="cheque">Cheque</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div>
                <Label>Date</Label>
                <Input type="date" value={salaryDate} onChange={(e) => setSalaryDate(e.target.value)} />
              </div>
              <div>
                <Label>Notes</Label>
                <Input value={salaryNotes} onChange={(e) => setSalaryNotes(e.target.value)} placeholder="Optional notes" />
              </div>
              <Button onClick={paySalary} className="w-full" disabled={!selectedSalaryEmployee}>
                Record Salary Payment
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>

      <Dialog open={isAddEmployeeOpen} onOpenChange={setIsAddEmployeeOpen}>
        <DialogContent className="max-w-md">
          <DialogHeader>
            <DialogTitle>Add Employee</DialogTitle>
          </DialogHeader>
          <div className="space-y-3">
            <div>
              <Label>Name</Label>
              <Input value={employeeName} onChange={(e) => setEmployeeName(e.target.value)} />
            </div>
            <div>
              <Label>Role</Label>
              <Input value={employeeRole} onChange={(e) => setEmployeeRole(e.target.value)} />
            </div>
            <div>
              <Label>Monthly Salary</Label>
              <Input type="number" value={employeeSalary} onChange={(e) => setEmployeeSalary(e.target.value)} />
            </div>
            <Button onClick={addEmployee} className="w-full">Save Employee</Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Filters */}
      <div className="flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold md:text-base">Filters</h3>
          <p className="text-xs text-muted-foreground">Open or collapse the date and mode filters.</p>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="rounded-xl"
          onClick={() => setFiltersOpen((prev) => !prev)}
        >
          {filtersOpen ? "Collapse Filters" : "Open Filters"}
        </Button>
      </div>

      {filtersOpen && (
      <Card>
        <CardContent className="p-3 md:p-4">
          <div className="flex flex-col gap-3 md:gap-4 md:flex-wrap md:flex-row md:items-end">
            <div className="flex-1 min-w-full md:min-w-[160px]">
              <Label className="text-xs md:text-sm">Start Date</Label>
              <Input
                type="date"
                value={startDate}
                onChange={(e) => { setStartDate(e.target.value); setPage(1) }}
                data-testid="start-date-filter"
                className="text-sm"
              />
            </div>
            <div className="flex-1 min-w-full md:min-w-[160px]">
              <Label className="text-xs md:text-sm">End Date</Label>
              <Input
                type="date"
                value={endDate}
                onChange={(e) => { setEndDate(e.target.value); setPage(1) }}
                data-testid="end-date-filter"
                className="text-sm"
              />
            </div>
            <div className="flex gap-1 md:gap-2 flex-wrap">
              <Button variant="outline" size="sm" onClick={() => setQuickDateFilter('today')} className="text-xs md:text-sm px-2 md:px-3 py-1 md:py-2">Today</Button>
              <Button variant="outline" size="sm" onClick={() => setQuickDateFilter('week')} className="text-xs md:text-sm px-2 md:px-3 py-1 md:py-2">Week</Button>
              <Button variant="outline" size="sm" onClick={() => setQuickDateFilter('month')} className="text-xs md:text-sm px-2 md:px-3 py-1 md:py-2">Month</Button>
              <Button variant="outline" size="sm" onClick={() => setQuickDateFilter('all')} className="text-xs md:text-sm px-2 md:px-3 py-1 md:py-2">All</Button>
            </div>
            {(tab === 'ledger' || tab === 'cashbook' || tab === 'bankbook') && (
              <div className="flex-1 min-w-full md:min-w-[160px]">
                <Label className="text-xs md:text-sm">Mode</Label>
                <Select value={paymentModeFilter} onValueChange={(val) => { setPaymentModeFilter(val); setPage(1) }}>
                  <SelectTrigger className="text-sm">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All</SelectItem>
                    <SelectItem value="cash">Cash</SelectItem>
                    <SelectItem value="bank">Bank</SelectItem>
                    <SelectItem value="upi">UPI</SelectItem>
                    <SelectItem value="cheque">Cheque</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
      )}

      {/* Tabs */}
      <Tabs value={tab} onValueChange={(val) => { setTab(val); setPage(1) }} className="space-y-4">
        <TabsList className="grid w-full grid-cols-3 md:grid-cols-5 lg:grid-cols-8 lg:w-auto h-auto">
          <TabsTrigger value="ledger" data-testid="ledger-tab" className="text-xs md:text-sm py-2 md:py-3">Ledger</TabsTrigger>
          <TabsTrigger value="cashbook" data-testid="cashbook-tab" className="text-xs md:text-sm py-2 md:py-3">Cashbook</TabsTrigger>
          <TabsTrigger value="bankbook" data-testid="bankbook-tab" className="text-xs md:text-sm py-2 md:py-3">Bankbook</TabsTrigger>
          <TabsTrigger value="gst" data-testid="gst-tab" className="text-xs md:text-sm py-2 md:py-3">GST</TabsTrigger>
          <TabsTrigger value="expenses" data-testid="expenses-tab" className="text-xs md:text-sm py-2 md:py-3">Expenses</TabsTrigger>
          <TabsTrigger value="salary" className="text-xs md:text-sm py-2 md:py-3">Salary</TabsTrigger>
          <TabsTrigger value="suppliers" className="text-xs md:text-sm py-2 md:py-3">Supplier Ledger</TabsTrigger>
          <TabsTrigger value="cheques" data-testid="cheques-tab" className="text-xs md:text-sm py-2 md:py-3">Cheques</TabsTrigger>
        </TabsList>

        {/* Ledger Tab */}
        <TabsContent value="ledger">
          <Card>
            <CardHeader className="py-3 md:py-4 px-3 md:px-6">
              <CardTitle className="text-base md:text-lg">General Ledger</CardTitle>
            </CardHeader>
            <CardContent className="px-2 md:px-6 py-3 md:py-4">
              <div className="overflow-x-auto rounded-lg border border-border">
                <table className="w-full text-xs md:text-sm">

                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Time</th>
                      <th className="text-left p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Type</th>
                      <th className="text-left p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300 hidden md:table-cell">Description</th>
                      <th className="text-right p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-red-300">Debit</th>
                      <th className="text-right p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-green-300">Credit</th>
                      <th className="text-right p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-purple-300">Balance</th>
                      <th className="text-left p-2 md:p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300 hidden lg:table-cell">Mode</th>
                    </tr>
                  </thead>

                  <tbody>
                    {ledger.data?.length > 0 ? (
                      ledger.data.map((entry, i) => {
                        const ledgerType = getLedgerTypeMeta(entry.type)

                        return (
                        <tr key={i} className="border-b border-border hover:bg-muted/30 transition">

                          <td className="p-2 md:p-3 text-xs md:text-sm">{formatDate(entry.date)}</td>
                          <td className="p-2 md:p-3 text-xs md:text-sm text-muted-foreground">{formatTime(entry.created_at)}</td>

                          <td className="p-2 md:p-3">
                            <span
                              className={`px-2 py-1 rounded text-xs font-medium ${ledgerType.className}`}
                            >
                              {ledgerType.label}
                            </span>
                          </td>

                          <td className="p-2 md:p-3 hidden md:table-cell truncate text-xs md:text-sm">{entry.description}</td>

                          <td className="p-2 md:p-3 text-right text-red-500 font-semibold text-xs md:text-sm whitespace-nowrap tabular-nums">
                            {entry.debit ? formatCurrency(entry.debit) : "-"}
                          </td>

                          <td className="p-2 md:p-3 text-right text-green-500 font-semibold text-xs md:text-sm whitespace-nowrap tabular-nums">
                            {entry.credit ? formatCurrency(entry.credit) : "-"}
                          </td>

                          <td
                            className={`p-2 md:p-3 text-right font-bold text-xs md:text-sm whitespace-nowrap tabular-nums ${entry.balance >= 0 ? "text-green-500" : "text-red-500"
                              }`}
                          >
                            {formatCurrency(entry.balance)}
                          </td>

                          <td className="p-2 md:p-3 hidden lg:table-cell">
                            <span className="px-2 py-1 bg-muted rounded text-xs">
                              {entry.mode}
                            </span>
                          </td>

                        </tr>
                      )})
                    ) : (
                      <tr>
                        <td colSpan="8" className="p-6 text-center text-muted-foreground text-sm">
                          No ledger entries found
                        </td>
                      </tr>
                    )}
                  </tbody>

                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Cashbook Tab */}
        <TabsContent value="cashbook">
          <Card>
            <CardHeader>
              <CardTitle>Cash Book</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Description</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-red-300">Debit</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-green-300">Credit</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-purple-300">Balance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cashbook.data && cashbook.data.length > 0 ? (
                      cashbook.data.map((entry, i) => (
                        <tr key={i} className="border-b hover:bg-muted/30">
                          <td className="p-3">{formatDate(entry.date)}</td>
                          <td className="p-3">{entry.description}</td>
                          <td className="p-3 text-right text-red-600 font-medium">
                            {entry.debit > 0 ? formatCurrency(entry.debit) : '-'}
                          </td>
                          <td className="p-3 text-right text-green-600 font-medium">
                            {entry.credit > 0 ? formatCurrency(entry.credit) : '-'}
                          </td>
                          <td className={`p-3 text-right font-bold ${entry.balance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatCurrency(entry.balance)}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="5" className="p-6 text-center text-gray-500">
                          No cash transactions found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Bankbook Tab */}
        <TabsContent value="bankbook">
          <Card>
            <CardHeader>
              <CardTitle>Bank Book</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Description</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Mode</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-red-300">Debit</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-green-300">Credit</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-purple-300">Balance</th>
                    </tr>
                  </thead>
                  <tbody>
                    {bankbook.data && bankbook.data.length > 0 ? (
                      bankbook.data.map((entry, i) => (
                        <tr key={i} className="border-b hover:bg-muted/30">
                          <td className="p-3">{formatDate(entry.date)}</td>
                          <td className="p-3">{entry.description}</td>
                          <td className="p-3">
                            <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                              {entry.mode}
                            </span>
                          </td>
                          <td className="p-3 text-right text-red-600 font-medium">
                            {entry.debit > 0 ? formatCurrency(entry.debit) : '-'}
                          </td>
                          <td className="p-3 text-right text-green-600 font-medium">
                            {entry.credit > 0 ? formatCurrency(entry.credit) : '-'}
                          </td>
                          <td className={`p-3 text-right font-bold ${entry.balance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                            {formatCurrency(entry.balance)}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="6" className="p-6 text-center text-gray-500">
                          No bank transactions found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* GST Tab */}
        <TabsContent value="gst">
          <Card>
            <CardHeader>
              <CardTitle>GST Ledger</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Invoice</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Customer</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-orange-300">Taxable</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-orange-300">CGST</th>
                      <th className="text-right p-3 font-semibold">SGST</th>
                      <th className="text-right p-3 font-semibold">IGST</th>
                      <th className="text-right p-3 font-semibold">Total GST</th>
                      <th className="text-right p-3 font-semibold">Total</th>
                    </tr>
                  </thead>
                  <tbody>
                    {gstLedger.data && gstLedger.data.length > 0 ? (
                      gstLedger.data.map((entry, i) => (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="p-3">{formatDate(entry.date)}</td>
                          <td className="p-3 font-medium">{entry.invoice}</td>
                          <td className="p-3">{entry.customer}</td>
                          <td className="p-3 text-right">{formatCurrency(entry.taxable_amount)}</td>
                          <td className="p-3 text-right text-blue-600">{formatCurrency(entry.cgst)}</td>
                          <td className="p-3 text-right text-purple-600">{formatCurrency(entry.sgst)}</td>
                          <td className="p-3 text-right text-orange-600">{formatCurrency(entry.igst)}</td>
                          <td className="p-3 text-right font-medium text-green-600">{formatCurrency(entry.gst_amount)}</td>
                          <td className="p-3 text-right font-bold">{formatCurrency(entry.total)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="9" className="p-6 text-center text-gray-500">
                          No GST entries found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Expenses Tab */}
        <TabsContent value="expenses">
          <Card>
            <CardHeader>
              <CardTitle>Expenses</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Type</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Title</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-red-300">Amount</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Mode</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Description</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Created By</th>
                    </tr>
                  </thead>
                  <tbody>
                    {expenses.data && expenses.data.length > 0 ? (
                      expenses.data.map((exp, i) => (
                        <tr key={i} className="border-b hover:bg-muted/30">
                          <td className="p-3">{formatDate(exp.expense_date)}</td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              exp.type === "salary_expense"
                                ? "bg-rose-100 text-rose-700"
                                : exp.type === "supplier_payment" || exp.type === "payment_out"
                                  ? "bg-orange-100 text-orange-700"
                                  : "bg-gray-100 text-gray-700"
                            }`}>
                              {exp.type_label || "Expense"}
                            </span>
                          </td>
                          <td className="p-3 font-medium">{exp.title}</td>
                          <td className="p-3 text-right font-bold text-red-600">{formatCurrency(exp.amount)}</td>
                          <td className="p-3">
                            <span className="px-2 py-1 bg-gray-100 rounded text-xs">
                              {exp.payment_mode}
                            </span>
                          </td>
                          <td className="p-3 text-gray-600">{exp.description || '-'}</td>
                          <td className="p-3">{exp.created_by_name || '-'}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="7" className="p-6 text-center text-gray-500">
                          No expenses found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="salary">
          <div className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Salary Entries</CardTitle>
                <p className="text-sm text-muted-foreground">
                  Clean salary history with date, type, employee, and pending status.
                </p>
              </CardHeader>
              <CardContent>
                <div className="relative mb-4">
                  <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    value={salaryHistoryLookup}
                    onChange={(e) => setSalaryHistoryLookup(e.target.value)}
                    placeholder="Search salary date, employee, role, note, or month"
                    className="pl-9"
                  />
                </div>
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-900 text-slate-100">
                      <tr>
                        <th className="p-3 text-left">Date</th>
                        <th className="p-3 text-left">Type</th>
                        <th className="p-3 text-left">Employee</th>
                        <th className="p-3 text-left">Role</th>
                        <th className="p-3 text-right">Salary Entry</th>
                        <th className="p-3 text-right">Paid</th>
                        <th className="p-3 text-right">Pending</th>
                        <th className="p-3 text-left">Mode</th>
                      </tr>
                    </thead>
                    <tbody>
                      {salaryEntries.length === 0 ? (
                        <tr>
                          <td colSpan="8" className="p-6 text-center text-muted-foreground">No salary entries found</td>
                        </tr>
                      ) : (
                        salaryEntries.map((entry) => {
                          const typeMeta = getSalaryEntryMeta(entry)

                          return (
                          <tr key={entry.id} className="border-b hover:bg-muted/30">
                            <td className="p-3">{formatDate(entry.payment_date || entry.created_at)}</td>
                            <td className="p-3">
                              <span className={`rounded-full px-2.5 py-1 text-xs font-semibold ${typeMeta.className}`}>
                                {typeMeta.label}
                              </span>
                            </td>
                            <td className="p-3 font-medium">
                              <div>{entry.employee_name || "-"}</div>
                              {entry.notes ? (
                                <div className="mt-1 text-xs text-muted-foreground">{entry.notes}</div>
                              ) : null}
                            </td>
                            <td className="p-3">{entry.employee_role || "-"}</td>
                            <td className="p-3 text-right">{formatCurrency(entry.total_salary)}</td>
                            <td className="p-3 text-right font-semibold text-green-600">{formatCurrency(entry.paid_amount)}</td>
                            <td className="p-3 text-right font-semibold text-red-600">{formatCurrency(entry.pending_amount)}</td>
                            <td className="p-3 uppercase">{entry.payment_mode || "-"}</td>
                          </tr>
                          )
                        })
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="power">
          <div className="space-y-4">
            <div className="flex flex-col gap-3 rounded-3xl border bg-muted/20 p-4 md:flex-row md:items-center md:justify-between">
              <div>
                <h3 className="text-base font-semibold">Power Bill History</h3>
                <p className="text-sm text-muted-foreground">Track all power bills without the extra summary cards.</p>
              </div>
              <Dialog open={isAddPowerBillOpen} onOpenChange={setIsAddPowerBillOpen}>
                <DialogTrigger asChild>
                  <Button className="gap-2 self-start md:self-auto">
                    <Receipt className="h-4 w-4" />
                    Add Power Bill
                  </Button>
                </DialogTrigger>
                <DialogContent>
                  <DialogHeader>
                    <DialogTitle>Add Power Bill</DialogTitle>
                  </DialogHeader>
                  <div className="space-y-3">
                    <div>
                      <Label>Amount</Label>
                      <Input type="number" value={powerBillAmount} onChange={(e) => setPowerBillAmount(e.target.value)} />
                    </div>
                    <div>
                      <Label>Payment Mode</Label>
                      <Select value={powerBillMode} onValueChange={setPowerBillMode}>
                        <SelectTrigger><SelectValue /></SelectTrigger>
                        <SelectContent>
                          <SelectItem value="cash">Cash</SelectItem>
                          <SelectItem value="bank">Bank</SelectItem>
                          <SelectItem value="upi">UPI</SelectItem>
                          <SelectItem value="cheque">Cheque</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label>Date</Label>
                      <Input type="date" value={powerBillDate} onChange={(e) => setPowerBillDate(e.target.value)} />
                    </div>
                    <div>
                      <Label>Description</Label>
                      <Input value={powerBillDescription} onChange={(e) => setPowerBillDescription(e.target.value)} />
                    </div>
                    <Button onClick={addPowerBill} className="w-full">Save Power Bill</Button>
                  </div>
                </DialogContent>
              </Dialog>
            </div>

            <Card>
              <CardHeader>
                <CardTitle>Power Bill History</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="overflow-x-auto rounded-lg border">
                  <table className="w-full text-sm">
                    <thead className="bg-slate-900 text-slate-100">
                      <tr>
                        <th className="p-3 text-left">Date</th>
                        <th className="p-3 text-left">Title</th>
                        <th className="p-3 text-left">Mode</th>
                        <th className="p-3 text-left">Description</th>
                        <th className="p-3 text-right">Amount</th>
                      </tr>
                    </thead>
                    <tbody>
                      {powerBills.length === 0 ? (
                        <tr>
                          <td colSpan="5" className="p-6 text-center text-muted-foreground">No power bills found</td>
                        </tr>
                      ) : (
                        powerBills.map((bill) => (
                          <tr key={bill.id} className="border-b hover:bg-muted/30">
                            <td className="p-3">{formatDate(bill.expense_date)}</td>
                            <td className="p-3 font-medium">{bill.title}</td>
                            <td className="p-3 capitalize">{bill.payment_mode}</td>
                            <td className="p-3">{bill.description || "-"}</td>
                            <td className="p-3 text-right font-semibold text-red-600">{formatCurrency(bill.amount)}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="suppliers">
          <div className="space-y-4">
            <Card className="overflow-hidden border border-white/10 bg-slate-950/65 shadow-[0_24px_60px_rgba(15,23,42,0.24)]">
              <CardHeader className="border-b border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(99,102,241,0.16),_transparent_28%),linear-gradient(135deg,#020617_0%,#0f172a_54%,#111827_100%)]">
                <CardTitle className="text-white">Supplier Ledger</CardTitle>
                <p className="text-sm text-slate-300">
                  Date and time based supplier ledger with dark theme styling.
                </p>
              </CardHeader>
              <CardContent className="p-4">
                {supplierLedgerEntries.length === 0 ? (
                  <div className="rounded-3xl border border-dashed border-white/10 bg-slate-950/40 p-10 text-center text-sm text-slate-400">
                    No supplier ledger entries found
                  </div>
                ) : (
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[1080px] text-sm">
                      <thead className="bg-slate-900/95 border-b border-white/10">
                        <tr className="text-slate-100">
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Supplier</th>
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Date</th>
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Time</th>
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Type</th>
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Bill</th>
                          <th className="p-3 text-right font-semibold uppercase tracking-wider text-xs text-rose-200">Debit</th>
                          <th className="p-3 text-right font-semibold uppercase tracking-wider text-xs text-emerald-200">Credit</th>
                          <th className="p-3 text-right font-semibold uppercase tracking-wider text-xs text-amber-200">Balance</th>
                          <th className="p-3 text-left font-semibold uppercase tracking-wider text-xs text-blue-200">Mode</th>
                        </tr>
                      </thead>
                      <tbody>
                        {supplierLedgerEntries.map((entry) => {
                          const typeMeta = getLedgerTypeMeta(entry.type)
                          const statusMeta = entry.status ? getSupplierBillStatusMeta(entry.status) : null

                          return (
                            <tr key={entry.id} className="border-b border-white/10 bg-slate-950/30 text-slate-100 hover:bg-slate-900/70">
                              <td className="p-3 align-top">
                                <div className="font-semibold text-white">{entry.supplier_name || "-"}</div>
                                <div className="text-xs text-slate-400">{entry.supplier_phone || entry.supplier_email || "No contact"}</div>
                              </td>
                              <td className="p-3 align-top text-slate-200">{formatDate(entry.date)}</td>
                              <td className="p-3 align-top text-slate-300">{formatTime(entry.created_at || entry.date)}</td>
                              <td className="p-3 align-top">
                                <span className={`inline-flex rounded-full px-2.5 py-1 text-xs font-semibold ${typeMeta.className}`}>
                                  {typeMeta.label}
                                </span>
                              </td>
                              <td className="p-3 align-top">
                                <div className="font-medium text-slate-100">{entry.bill_number || "Direct Entry"}</div>
                                {statusMeta && (
                                  <span className={`mt-2 inline-flex rounded-full px-2.5 py-1 text-[11px] font-semibold ${statusMeta.className}`}>
                                    {statusMeta.label}
                                  </span>
                                )}
                              </td>
                              <td className="p-3 align-top text-right font-semibold text-rose-600 tabular-nums">
                                {Number(entry.debit || 0) > 0 ? formatCurrency(entry.debit) : "-"}
                              </td>
                              <td className="p-3 align-top text-right font-semibold text-emerald-600 tabular-nums">
                                {Number(entry.credit || 0) > 0 ? formatCurrency(entry.credit) : "-"}
                              </td>
                              <td className={`p-3 align-top text-right font-bold tabular-nums ${Number(entry.balance || 0) >= 0 ? "text-amber-300" : "text-emerald-300"}`}>
                                {formatCurrency(entry.balance || 0)}
                              </td>
                              <td className="p-3 align-top uppercase text-slate-300">{entry.mode || "-"}</td>
                            </tr>
                          )
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        {/* Cheques Tab */}
        <TabsContent value="cheques">
          <Card>
            <CardHeader>
              <CardTitle>Cheque Management</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gradient-to-r from-slate-900 via-slate-800 to-slate-900 border-b-2 border-blue-500/50 sticky top-0 z-10 shadow-lg">
                    <tr className="text-slate-100">
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Cheque No.</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Date</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Party</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Type</th>
                      <th className="text-right p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-purple-300">Amount</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Bank</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-green-300">Status</th>
                      <th className="text-left p-3 font-bold text-xs md:text-sm uppercase tracking-wider text-blue-300">Cleared Date</th>
                    </tr>
                  </thead>
                  <tbody>
                    {cheques.data && cheques.data.length > 0 ? (
                      cheques.data.map((cheque, i) => (
                        <tr key={i} className="border-b hover:bg-muted/30">
                          <td className="p-3 font-medium">{cheque.cheque_number}</td>
                          <td className="p-3">{formatDate(cheque.cheque_date)}</td>
                          <td className="p-3">{cheque.party_name}</td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs ${cheque.party_type === 'customer' ? 'bg-green-100 text-green-800' : 'bg-orange-100 text-orange-800'
                              }`}>
                              {cheque.party_type}
                            </span>
                          </td>
                          <td className="p-3 text-right font-bold">{formatCurrency(cheque.amount)}</td>
                          <td className="p-3">{cheque.bank_name || '-'}</td>
                          <td className="p-3">
                            <span className={`px-2 py-1 rounded text-xs font-medium ${cheque.status === 'cleared' ? 'bg-green-100 text-green-800' :
                              cheque.status === 'bounced' ? 'bg-red-100 text-red-800' :
                                'bg-yellow-100 text-yellow-800'
                              }`}>
                              {cheque.status}
                            </span>
                          </td>
                          <td className="p-3">{formatDate(cheque.cleared_date)}</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan="8" className="p-6 text-center text-gray-500">
                          No cheques found
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Pagination */}
      {currentData.pagination && currentData.pagination.total_pages > 1 && (
        <Card>
          <CardContent className="p-3 md:p-4">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
              <p className="text-xs md:text-sm text-gray-600 break-words">
                Page <span className="font-semibold">{currentData.pagination.page}</span> of <span className="font-semibold">{currentData.pagination.total_pages}</span> ({currentData.pagination.total} entries)
              </p>
              <div className="flex gap-2 w-full sm:w-auto">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page === 1}
                  onClick={() => setPage(page - 1)}
                  data-testid="prev-page-btn"
                  className="flex-1 sm:flex-none text-xs md:text-sm"
                >
                  <ChevronLeft className="w-3 md:w-4 h-3 md:h-4" />
                  <span className="hidden sm:inline">Previous</span>
                  <span className="sm:hidden">Prev</span>
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={page >= currentData.pagination.total_pages}
                  onClick={() => setPage(page + 1)}
                  data-testid="next-page-btn"
                  className="flex-1 sm:flex-none text-xs md:text-sm"
                >
                  <span className="hidden sm:inline">Next</span>
                  <span className="sm:hidden">Next</span>
                  <ChevronRight className="w-3 md:w-4 h-3 md:h-4" />
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

    </div>
  )
}

