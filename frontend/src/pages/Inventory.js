"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import axios from "axios"
import { useSearchParams } from "react-router-dom"
import { toast } from "sonner"
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  Download,
  Filter,
  History,
  Loader2,
  PackageSearch,
  Search,
} from "lucide-react"
import { Capacitor } from "@capacitor/core"
import {
  initializePdfDownloadSupport,
  savePdfWithNotification,
} from "@/lib/pdfHandler"

import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
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

const API =
  `${process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"}/api`

const currency = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 2,
})

const getAuthHeaders = () => {
  const token = localStorage.getItem("token")
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const tabButtonClass = isActive =>
  isActive ? "bg-primary text-primary-foreground" : ""

const isNativePlatform = () => Capacitor.isNativePlatform()

const floatingCardClass =
  "rounded-2xl border border-white/10 bg-slate-950/35 backdrop-blur-md shadow-[0_18px_40px_rgba(15,23,42,0.18)]"

const floatingInnerPanelClass =
  "rounded-2xl border border-white/10 bg-white/[0.04] backdrop-blur-sm shadow-[0_14px_28px_rgba(15,23,42,0.14)]"

const floatingSoftRowClass =
  "rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-sm"

export default function Inventory() {
  const [activeTab, setActiveTab] = useState("report")

  const [skuInput, setSkuInput] = useState("")
  const [lookupData, setLookupData] = useState(null)
  const [qtyInputs, setQtyInputs] = useState({})
  const [reason, setReason] = useState("")
  const [loading, setLoading] = useState(false)
  const [reportLoading, setReportLoading] = useState(false)
  const [downloadingPdf, setDownloadingPdf] = useState(false)
  const [transactionsLoading, setTransactionsLoading] = useState(false)
  const skuRef = useRef(null)

  const [transactions, setTransactions] = useState([])
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState("all")
  const [page, setPage] = useState(1)
  const limit = 30
  const [total, setTotal] = useState(0)
  const [searchParams] = useSearchParams()

  const [categories, setCategories] = useState([])
  const [reportSearch, setReportSearch] = useState("")
  const [selectedCategory, setSelectedCategory] = useState("all")
  const [activeCategory, setActiveCategory] = useState(null)
  const [showMobileFilters, setShowMobileFilters] = useState(false)
  const [inventoryReport, setInventoryReport] = useState({
    summary: {
      total_categories: 0,
      total_products: 0,
      total_stock: 0,
      low_stock_products: 0,
    },
    categories: [],
  })

  useEffect(() => {
    initializePdfDownloadSupport().catch(() => {
      // keep export working even if notifications are unavailable
    })
  }, [])

  const fetchCategories = useCallback(async () => {
    try {
      const res = await axios.get(`${API}/categories`, {
        headers: getAuthHeaders(),
      })
      setCategories(res.data || [])
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to load categories")
    }
  }, [])

  const fetchInventoryReport = useCallback(async () => {
    try {
      setReportLoading(true)
      const params = {}

      if (reportSearch.trim()) params.search = reportSearch.trim()
      if (selectedCategory !== "all") params.category_id = selectedCategory

      const res = await axios.get(`${API}/inventory/report`, {
        params,
        headers: getAuthHeaders(),
      })

      setInventoryReport(res.data)
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to load inventory report")
    } finally {
      setReportLoading(false)
    }
  }, [reportSearch, selectedCategory])
const downloadInventoryPdf = async () => {
  try {
    setDownloadingPdf(true)

    const params = {}
    if (reportSearch.trim()) params.search = reportSearch.trim()
    if (selectedCategory !== "all") params.category_id = selectedCategory

    const response = await axios.get(`${API}/inventory/report/pdf`, {
      params,
      headers: getAuthHeaders(),
      responseType: "blob",
    })

    const fileName = `inventory_${Date.now()}.pdf`

    const downloadResult = await savePdfWithNotification({
      blob: response.data,
      fileName,
    })

    toast.success(
      isNativePlatform()
        ? (
          downloadResult.notificationsGranted
            ? "Inventory PDF saved. Check your notification."
            : "Inventory PDF saved to Files. Notification permission denied."
        )
        : "Inventory PDF downloaded"
    )

  } catch (error) {
    console.error(error)
    toast.error("Failed to download PDF")
  } finally {
    setDownloadingPdf(false)
  }
}


  const lookupSku = useCallback(async skuOverride => {
    const sku = (skuOverride ?? skuInput).trim()

    if (!sku) {
      toast.error("Enter SKU")
      return
    }

    try {
      setLoading(true)
      const res = await axios.get(`${API}/inventory/lookup/${sku}`, {
        headers: getAuthHeaders(),
      })

      setLookupData(res.data)

      const map = {}
      if (res.data.variants?.length > 0) {
        res.data.variants.forEach(variant => {
          map[variant.v_sku] = ""
        })
      } else {
        map[res.data.parent_sku] = ""
      }

      setQtyInputs(map)
    } catch (error) {
      toast.error(error.response?.data?.detail || "SKU not found")
      setLookupData(null)
    } finally {
      setLoading(false)
    }
  }, [skuInput])

  const handleQtyChange = (sku, value) => {
    setQtyInputs(prev => ({ ...prev, [sku]: value }))
  }

  const resetForm = () => {
    setSkuInput("")
    setLookupData(null)
    setQtyInputs({})
    setReason("")
    skuRef.current?.focus()
  }

  const submitInward = async () => {
    try {
      setLoading(true)
      const payloads = Object.entries(qtyInputs)
        .filter(([, quantity]) => Number(quantity) > 0)
        .map(([sku, quantity]) =>
          axios.post(
            `${API}/inventory/material-inward/sku`,
            { sku, quantity: Number(quantity) },
            { headers: getAuthHeaders() }
          )
        )

      if (!payloads.length) {
        toast.error("Enter quantity")
        return
      }

      await Promise.all(payloads)
      toast.success("Stock added successfully")
      resetForm()
      fetchInventoryReport()
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed")
    } finally {
      setLoading(false)
    }
  }

  const submitOutward = async () => {
    if (!reason.trim()) {
      toast.error("Reason required")
      return
    }

    try {
      setLoading(true)
      const payloads = Object.entries(qtyInputs)
        .filter(([, quantity]) => Number(quantity) > 0)
        .map(([sku, quantity]) =>
          axios.post(
            `${API}/inventory/material-outward/sku`,
            { sku, quantity: Number(quantity), reason: reason.trim() },
            { headers: getAuthHeaders() }
          )
        )

      if (!payloads.length) {
        toast.error("Enter quantity")
        return
      }

      await Promise.all(payloads)
      toast.success("Stock deducted successfully")
      resetForm()
      fetchInventoryReport()
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed")
    } finally {
      setLoading(false)
    }
  }

  const fetchTransactions = useCallback(async () => {
    try {
      setTransactionsLoading(true)
      const params = { page, limit }
      if (filterType !== "all") params.type = filterType

      const res = await axios.get(`${API}/inventory/transactions`, {
        params,
        headers: getAuthHeaders(),
      })

      setTransactions(res.data.data || [])
      setTotal(res.data.total || 0)
    } catch {
      toast.error("Failed to load transactions")
    } finally {
      setTransactionsLoading(false)
    }
  }, [filterType, limit, page])

  useEffect(() => {
    fetchCategories()
  }, [fetchCategories])

  useEffect(() => {
    if (activeTab === "history") {
      fetchTransactions()
    }
  }, [activeTab, fetchTransactions])

  useEffect(() => {
    const timer = setTimeout(() => {
      fetchInventoryReport()
    }, 300)

    return () => clearTimeout(timer)
  }, [fetchInventoryReport])

  useEffect(() => {
    const sku = searchParams.get("sku")
    const mode = searchParams.get("mode")

    if (sku) {
      setSkuInput(sku)
      setActiveTab(mode === "out" ? "outward" : "inward")
      lookupSku(sku)
    }
  }, [lookupSku, searchParams])

  const filteredTransactions = useMemo(() => {
    if (!searchTerm.trim()) return transactions
    const query = searchTerm.toLowerCase()

    return transactions.filter(transaction =>
      transaction.product_name?.toLowerCase().includes(query) ||
      transaction.product_code?.toLowerCase().includes(query) ||
      transaction.variant_sku?.toLowerCase().includes(query)
    )
  }, [transactions, searchTerm])

  const totalPages = Math.ceil(total / limit)
  const reportCategories = useMemo(
    () => inventoryReport.categories || [],
    [inventoryReport.categories]
  )
  const summary = inventoryReport.summary || {}
  const activeCategoryData = useMemo(
    () => reportCategories.find(category => category.category_id === activeCategory) || null,
    [activeCategory, reportCategories]
  )

  return (
    <div className="space-y-5 bg-[#050816] p-3 text-slate-100 md:p-5">
      <div className="space-y-4">
        <div className="flex items-center justify-between gap-2">
          <div className="min-w-0">
            <h1 className="hidden text-2xl font-bold tracking-tight text-white md:block md:text-3xl">
              Inventory 
            </h1>
            <p className="mt-1 hidden text-sm text-slate-300 md:block">
              Compact category-first stock analysis for faster inventory checks.
            </p>
          </div>

         <Button
  onClick={downloadInventoryPdf}
  disabled={downloadingPdf}
  className="hidden md:flex h-9 rounded-xl bg-[#4f46e5] px-3 text-xs font-semibold text-white shadow-lg shadow-indigo-950/40 hover:bg-[#4338ca] md:h-10 md:px-4 md:text-sm"
>
  {downloadingPdf ? (
    <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin md:mr-2 md:h-4 md:w-4" />
  ) : (
    <Download className="mr-1.5 h-3.5 w-3.5 md:mr-2 md:h-4 md:w-4" />
  )}
  <span className="hidden md:inline">Download PDF</span>
</Button>
        </div>

        <div className="grid grid-cols-3 gap-2">
          <SummaryCard title="Categories" mobileTitle="Cat" value={summary.total_categories || 0} />
          <SummaryCard title="Products" mobileTitle="Prod" value={summary.total_products || 0} />
          <SummaryCard title="Total Stock" mobileTitle="Stock" value={summary.total_stock || 0} />
        </div>

        <div className="flex items-center gap-2 overflow-x-auto pb-1">
          <Button
            variant={activeTab === "report" ? "default" : "outline"}
            className={`h-9 shrink-0 rounded-xl border-slate-700 px-3 text-xs ${tabButtonClass(activeTab === "report")}`}
            onClick={() => setActiveTab("report")}
          >
            <PackageSearch className="h-3.5 w-3.5 md:mr-1.5" />
            <span className="hidden md:inline">Stock Report</span>
          </Button>

          <Button
            variant={activeTab === "inward" ? "default" : "outline"}
            className={`h-9 shrink-0 rounded-xl border-slate-700 px-3 text-xs ${tabButtonClass(activeTab === "inward")}`}
            onClick={() => setActiveTab("inward")}
          >
            <ArrowDownToLine className="h-3.5 w-3.5 md:mr-1.5" />
            <span className="hidden md:inline">Inward</span>
          </Button>

          <Button
            variant={activeTab === "outward" ? "default" : "outline"}
            className={`h-9 shrink-0 rounded-xl border-slate-700 px-3 text-xs ${tabButtonClass(activeTab === "outward")}`}
            onClick={() => setActiveTab("outward")}
          >
            <ArrowUpFromLine className="h-3.5 w-3.5 md:mr-1.5" />
            <span className="hidden md:inline">Outward</span>
          </Button>

          <Button
            variant={activeTab === "history" ? "default" : "outline"}
            className={`h-9 shrink-0 rounded-xl border-slate-700 px-3 text-xs ${tabButtonClass(activeTab === "history")}`}
            onClick={() => setActiveTab("history")}
          >
            <History className="h-3.5 w-3.5 md:mr-1.5" />
            <span className="hidden md:inline">History</span>
          </Button>
           <Button
            onClick={downloadInventoryPdf}
            disabled={downloadingPdf}
            className="h-9 rounded-xl bg-[#4f46e5] px-3 text-xs font-semibold text-white shadow-lg shadow-indigo-950/40 hover:bg-[#4338ca] md:h-10 md:px-4 md:text-sm"
          >
            {downloadingPdf ? (
              <Loader2 className="mr-1.5 h-3.5 w-3.5 animate-spin md:mr-2 md:h-4 md:w-4" />
            ) : (
              <Download className="mr-1.5 h-3.5 w-3.5 md:mr-2 md:h-4 md:w-4" />
            )}
            <span className="hidden md:inline">Download PDF</span>
          </Button>
        </div>
      </div>

      {activeTab === "report" && (
        <div className="space-y-4">
          <Card className={floatingCardClass}>
            <CardContent className="space-y-2 p-3">
              <div className="flex items-center gap-2">
                <div className="relative">
                  <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
                  <Input
                    value={reportSearch}
                    onChange={event => setReportSearch(event.target.value)}
                    placeholder="Search by product name, code, SKU, or category"
                    className="h-10 rounded-xl border-slate-700 bg-[#0f172f] pl-9 text-sm text-white placeholder:text-slate-400"
                  />
                </div>

                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setShowMobileFilters(current => !current)}
                  className="h-10 shrink-0 rounded-xl border-slate-700 bg-[#0f172f] px-3 text-xs text-white md:hidden"
                >
                  <Filter className="mr-1.5 h-3.5 w-3.5" />
                  Filter
                </Button>

                <div className="hidden md:block md:w-[210px]">
                  <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                    <SelectTrigger className="h-10 rounded-xl border-slate-700 bg-[#0f172f] text-white">
                      <SelectValue placeholder="Filter by category" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Categories</SelectItem>
                      {categories.map(category => (
                        <SelectItem key={category.id} value={category.id}>
                          {category.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {showMobileFilters && (
                <div className="md:hidden">
                  <Select value={selectedCategory} onValueChange={setSelectedCategory}>
                    <SelectTrigger className="h-10 rounded-xl border-slate-700 bg-[#0f172f] text-white">
                      <SelectValue placeholder="Filter by category" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="all">All Categories</SelectItem>
                      {categories.map(category => (
                        <SelectItem key={category.id} value={category.id}>
                          {category.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              )}
            </CardContent>
          </Card>

          {reportLoading ? (
            <Card className={floatingCardClass}>
              <CardContent className="flex min-h-[220px] items-center justify-center">
                <div className="flex items-center gap-3 text-sm text-slate-300">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Loading inventory report...
                </div>
              </CardContent>
            </Card>
          ) : reportCategories.length === 0 ? (
            <Card className={floatingCardClass}>
              <CardContent className="flex min-h-[220px] flex-col items-center justify-center gap-2 text-center">
                <PackageSearch className="h-10 w-10 text-slate-300" />
                <div className="text-base font-semibold text-white">No products found</div>
                <p className="max-w-md text-sm text-slate-300">
                  Try changing the search text or category filter to view more inventory items.
                </p>
              </CardContent>
            </Card>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {reportCategories.map(category => (
                <button
                  key={category.category_id}
                  type="button"
                  onClick={() => setActiveCategory(category.category_id)}
                  className="group rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.52),rgba(15,23,42,0.26))] p-2.5 text-left shadow-[0_16px_30px_rgba(15,23,42,0.16)] backdrop-blur-md transition hover:border-indigo-500/50 hover:shadow-[0_20px_36px_rgba(79,70,229,0.14)]"
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <h2 className="truncate text-[13px] font-bold text-white md:text-base">
                        {category.category_name}
                      </h2>
                      <p className="mt-0.5 text-[11px] text-slate-400">
                        {category.product_count} products
                      </p>
                    </div>

                    <span className="rounded-full bg-indigo-500/15 px-2 py-1 text-[10px] font-semibold text-indigo-200">
                      Open
                    </span>
                  </div>

                  <div className="mt-2.5 grid grid-cols-3 gap-1.5">
                    <CompactBadge label="Products" value={category.product_count} />
                    <CompactBadge label="Stock" value={category.total_stock} highlight="indigo" />
                    <CompactBadge label="Low" value={category.low_stock_count} highlight="red" />
                  </div>

                  <div className={`mt-2.5 flex items-center justify-between px-2 py-1.5 ${floatingSoftRowClass}`}>
                    <span className="text-[10px] font-medium text-slate-400">
                      Largest categories first
                    </span>
                    <span className="text-[11px] font-semibold text-slate-200 group-hover:text-white">
                      View Details
                    </span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {(activeTab === "inward" || activeTab === "outward") && (
        <Card className={`max-w-3xl ${floatingCardClass}`}>
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-white">
              {activeTab === "inward" ? "Material Inward" : "Material Outward"}
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label>Scan / Enter SKU</Label>
              <Input
                ref={skuRef}
                value={skuInput}
                onChange={event => setSkuInput(event.target.value)}
                onKeyDown={event => event.key === "Enter" && lookupSku()}
                placeholder="Parent SKU or Variant SKU"
                className="h-11 rounded-xl border-slate-700 bg-[#0f172f] text-white"
              />
            </div>

            {lookupData && (
              <div className={`space-y-3 p-4 ${floatingInnerPanelClass}`}>
                <div>
                  <div className="text-lg font-semibold text-white">{lookupData.product_name}</div>
                  <div className="text-sm text-slate-300">
                    Parent SKU: <span className="font-mono">{lookupData.parent_sku}</span>
                  </div>
                  <div className="mt-1 text-sm font-semibold text-slate-100">
                    Total Stock: {lookupData.total_stock}
                  </div>
                </div>

                {(lookupData.variants?.length > 0
                  ? lookupData.variants
                  : [{ v_sku: lookupData.parent_sku, stock: lookupData.total_stock }]
                ).map(variant => (
                  <div
                    key={variant.v_sku}
                    className={`space-y-3 p-4 ${floatingInnerPanelClass}`}
                  >
                    <div className="flex items-center justify-between gap-4">
                      <span className="font-mono text-sm font-semibold text-slate-100">
                        {variant.v_sku}
                      </span>
                      <span className="text-sm font-semibold text-slate-100">
                        Stock: {variant.stock}
                      </span>
                    </div>

                    <Input
                      type="number"
                      min="1"
                      placeholder="Qty"
                      value={qtyInputs[variant.v_sku] || ""}
                      onChange={event => handleQtyChange(variant.v_sku, event.target.value)}
                      className="h-11 rounded-xl border-slate-700 bg-[#111936] text-white"
                    />
                  </div>
                ))}
              </div>
            )}

            {activeTab === "outward" && (
              <div className="space-y-2">
                <Label>Reason</Label>
                <Input
                  value={reason}
                  onChange={event => setReason(event.target.value)}
                  className="h-11 rounded-xl border-slate-700 bg-[#0f172f] text-white"
                  placeholder="Damaged / Sold / Returned"
                />
              </div>
            )}

            <Button
              className="h-11 w-full rounded-xl bg-[#4f46e5] text-white hover:bg-[#4338ca]"
              disabled={loading}
              onClick={activeTab === "inward" ? submitInward : submitOutward}
            >
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Submit
            </Button>
          </CardContent>
        </Card>
      )}

      {activeTab === "history" && (
        <Card className={floatingCardClass}>
          <CardHeader>
            <CardTitle className="text-lg font-semibold text-white">
              Transaction History
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex flex-col gap-3 md:flex-row">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3.5 h-4 w-4 text-slate-400" />
                <Input
                  className="h-11 rounded-xl border-slate-700 bg-[#0f172f] pl-9 text-white"
                  placeholder="Search product / SKU"
                  value={searchTerm}
                  onChange={event => setSearchTerm(event.target.value)}
                />
              </div>

              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="h-11 w-full rounded-xl border-slate-700 bg-[#0f172f] text-white md:w-40">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All</SelectItem>
                  <SelectItem value="IN">Inward</SelectItem>
                  <SelectItem value="OUT">Outward</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700 text-left text-slate-300">
                    <th className="px-3 py-3 font-semibold">Date</th>
                    <th className="px-3 py-3 font-semibold">Type</th>
                    <th className="px-3 py-3 font-semibold">Product</th>
                    <th className="px-3 py-3 font-semibold">Code</th>
                    <th className="px-3 py-3 font-semibold">V-SKU</th>
                    <th className="px-3 py-3 font-semibold">Qty</th>
                    <th className="px-3 py-3 font-semibold">Stock After</th>
                    <th className="px-3 py-3 font-semibold">By</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredTransactions.map(transaction => (
                    <tr key={transaction.id} className="border-b border-white/10 last:border-b-0">
                      <td className="px-3 py-3">{new Date(transaction.created_at).toLocaleString()}</td>
                      <td
                        className={`px-3 py-3 font-semibold ${
                          transaction.type === "IN" ? "text-green-600" : "text-red-600"
                        }`}
                      >
                        {transaction.type}
                      </td>
                      <td className="px-3 py-3">{transaction.product_name}</td>
                      <td className="px-3 py-3 font-mono">{transaction.product_code}</td>
                      <td className="px-3 py-3 font-mono">{transaction.variant_sku || "-"}</td>
                      <td className="px-3 py-3">{transaction.quantity}</td>
                      <td className="px-3 py-3 font-semibold">
                        {transaction.variant_sku
                          ? transaction.variant_stock_after ?? "-"
                          : transaction.stock_after}
                      </td>
                      <td className="px-3 py-3">{transaction.created_by || "-"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {transactionsLoading && (
              <div className="flex items-center justify-center py-4 text-sm text-slate-500">
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                Loading history...
              </div>
            )}

            <div className="flex items-center justify-between">
              <Button disabled={page === 1} onClick={() => setPage(current => current - 1)}>
                Prev
              </Button>

              <span className="text-sm text-slate-300">
                Page {page} / {totalPages || 1}
              </span>

              <Button
                disabled={page >= totalPages}
                onClick={() => setPage(current => current + 1)}
              >
                Next
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      <Dialog
        open={Boolean(activeCategoryData)}
        onOpenChange={open => {
          if (!open) setActiveCategory(null)
        }}
      >
        <DialogContent className="max-h-[92vh] max-w-4xl overflow-hidden rounded-2xl border border-white/10 bg-[#060b1a]/95 p-0 text-slate-100 backdrop-blur-xl">
          {activeCategoryData && (
            <>
              <DialogHeader className="border-b border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.72),rgba(15,23,42,0.38))] px-4 py-4 backdrop-blur-md sm:px-5">
                <DialogTitle className="text-left text-base font-bold text-white sm:text-lg">
                  {activeCategoryData.category_name}
                </DialogTitle>
                <div className="mt-2 flex flex-wrap gap-2">
                  <CategoryBadge label="Products" value={activeCategoryData.product_count} />
                  <CategoryBadge label="Total Stock" value={activeCategoryData.total_stock} highlight="indigo" />
                  <CategoryBadge label="Low Stock" value={activeCategoryData.low_stock_count} highlight="red" />
                </div>
              </DialogHeader>

              <div className="max-h-[70vh] overflow-y-auto px-3 py-3 sm:px-5">
                {activeCategoryData.products.length === 0 ? (
                  <div className="rounded-2xl border border-dashed border-white/10 bg-slate-950/25 p-6 text-center text-sm text-slate-300 backdrop-blur-sm">
                    No products in this category.
                  </div>
                ) : (
                  <>
                    <div className="space-y-1.5 md:hidden">
                      {activeCategoryData.products.map(product => (
                        <div
                          key={product.id}
                          className="rounded-2xl border border-white/10 bg-slate-950/30 p-2.5 shadow-[0_14px_26px_rgba(15,23,42,0.14)] backdrop-blur-sm"
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="truncate text-[13px] font-semibold text-white">
                                {product.name}
                              </div>
                              <div className="mt-1 text-[11px] text-slate-400">
                                {product.product_code} • <span className="font-mono">{product.sku}</span>
                              </div>
                            </div>
                            <span className={`rounded-full border px-2 py-1 text-[10px] font-semibold ${stockToneFor(product.is_low_stock)}`}>
                              {product.stock}
                            </span>
                          </div>

                          <div className="mt-2 grid grid-cols-3 gap-1.5">
                            <MiniStat label="Min" value={product.min_stock} />
                            <MiniStat
                              label="Stock"
                              value={product.stock}
                              className={product.is_low_stock ? "text-red-400" : "text-emerald-400"}
                            />
                            <MiniStat label="Price" value={currency.format(product.selling_price || 0)} />
                          </div>
                        </div>
                      ))}
                    </div>

                    <div className="hidden overflow-hidden rounded-2xl border border-white/10 bg-slate-950/30 backdrop-blur-sm md:block">
                      <table className="min-w-full text-left text-sm">
                        <thead className="bg-white/[0.04] text-slate-200">
                          <tr>
                            <th className="px-4 py-3 font-semibold">Product Name</th>
                            <th className="px-4 py-3 font-semibold">Code</th>
                            <th className="px-4 py-3 font-semibold">SKU</th>
                            <th className="px-4 py-3 font-semibold">Stock</th>
                            <th className="px-4 py-3 font-semibold">Min</th>
                            <th className="px-4 py-3 font-semibold">Price</th>
                          </tr>
                        </thead>
                        <tbody>
                          {activeCategoryData.products.map(product => (
                            <tr key={product.id} className="border-t border-white/10 bg-transparent">
                              <td className="px-4 py-3 font-medium text-white">{product.name}</td>
                              <td className="px-4 py-3 font-mono text-slate-300">{product.product_code}</td>
                              <td className="px-4 py-3 font-mono text-slate-300">{product.sku}</td>
                              <td className={`px-4 py-3 font-bold ${product.is_low_stock ? "text-red-400" : "text-emerald-400"}`}>
                                {product.stock}
                              </td>
                              <td className={`px-4 py-3 font-semibold ${product.is_low_stock ? "text-red-300" : "text-slate-200"}`}>
                                {product.min_stock}
                              </td>
                              <td className="px-4 py-3 font-semibold text-slate-100">
                                {currency.format(product.selling_price || 0)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  )
}

function SummaryCard({ title, mobileTitle, value, danger = false }) {
  return (
    <Card className="rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.6),rgba(15,23,42,0.28))] shadow-[0_18px_36px_rgba(15,23,42,0.16)] backdrop-blur-md">
      <CardContent className="p-2.5 md:p-4">
        <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-slate-400 md:text-xs md:tracking-[0.18em]">
          <span className="md:hidden">{mobileTitle || title}</span>
          <span className="hidden md:inline">{title}</span>
        </p>
        <p className={`mt-1.5 text-base font-bold md:text-2xl ${danger ? "text-[#ff5c7a]" : "text-white"}`}>
          {value}
        </p>
      </CardContent>
    </Card>
  )
}

function CategoryBadge({ label, value, danger = false, highlight = "default" }) {
  const tone =
    highlight === "red" || danger
      ? "border-red-500/30 bg-red-500/10 text-red-200"
      : highlight === "indigo"
      ? "border-indigo-500/30 bg-indigo-500/15 text-indigo-100"
      : "border-white/10 bg-white/[0.04] text-slate-100"

  return (
    <div className={`rounded-full border px-3 py-1.5 text-xs font-semibold ${tone}`}>
      {label}: {value}
    </div>
  )
}

function CompactBadge({ label, value, highlight = "default" }) {
  const tone =
    highlight === "red"
      ? "border-red-500/20 bg-red-500/10 text-red-200"
      : highlight === "indigo"
      ? "border-indigo-500/20 bg-indigo-500/10 text-indigo-100"
      : "border-white/10 bg-white/[0.04] text-slate-100"

  return (
    <div className={`rounded-xl border px-2.5 py-2 ${tone}`}>
      <div className="text-[9px] uppercase tracking-[0.12em] text-slate-400">{label}</div>
      <div className="mt-0.5 text-xs font-bold">{value}</div>
    </div>
  )
}

function MiniStat({ label, value, className = "" }) {
  return (
    <div className="rounded-xl border border-white/10 bg-white/[0.04] px-2 py-1.5 backdrop-blur-sm">
      <div className="text-[10px] uppercase tracking-[0.16em] text-slate-400">{label}</div>
      <div className={`mt-1 truncate text-xs font-semibold text-slate-100 ${className}`}>
        {value}
      </div>
    </div>
  )
}

function stockToneFor(isLowStock) {
  return isLowStock
    ? "border-red-500/30 bg-red-500/10 text-red-200"
    : "border-emerald-500/30 bg-emerald-500/10 text-emerald-200"
}
