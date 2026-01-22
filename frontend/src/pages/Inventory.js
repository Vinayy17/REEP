"use client"

import { useEffect, useRef, useState, useMemo } from "react"
import axios from "axios"
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
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { toast } from "sonner"
import {
  ArrowDownToLine,
  ArrowUpFromLine,
  History,
  Search,
} from "lucide-react"

const API =
  `${process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"}/api`

export default function Inventory() {
  const [activeTab, setActiveTab] = useState("inward")

  // ================= SKU LOOKUP =================
  const [skuInput, setSkuInput] = useState("")
  const [lookupData, setLookupData] = useState(null)
  const [qtyInputs, setQtyInputs] = useState({})
  const [reason, setReason] = useState("")
  const [loading, setLoading] = useState(false)
  const skuRef = useRef(null)

  // ================= HISTORY =================
  const [transactions, setTransactions] = useState([])
  const [searchTerm, setSearchTerm] = useState("")
  const [filterType, setFilterType] = useState("all")
  const [page, setPage] = useState(1)
  const limit = 30
  const [total, setTotal] = useState(0)

  // ================= LOOKUP =================
  const lookupSku = async () => {
    if (!skuInput.trim()) {
      toast.error("Enter SKU")
      return
    }

    try {
      setLoading(true)
      const res = await axios.get(
        `${API}/inventory/lookup/${skuInput.trim()}`
      )

      setLookupData(res.data)

      const map = {}

      // 🔥 VARIANTS EXIST
      if (res.data.variants?.length > 0) {
        res.data.variants.forEach(v => {
          map[v.v_sku] = ""
        })
      }
      // 🔥 NO VARIANTS → PRODUCT LEVEL
      else {
        map[res.data.parent_sku] = ""
      }

      setQtyInputs(map)
    } catch (e) {
      toast.error(e.response?.data?.detail || "SKU not found")
      setLookupData(null)
    } finally {
      setLoading(false)
    }
  }

  // ================= QTY =================
  const handleQtyChange = (sku, value) => {
    setQtyInputs(prev => ({ ...prev, [sku]: value }))
  }

  // ================= INWARD =================
  const submitInward = async () => {
    try {
      setLoading(true)

      const reqs = Object.entries(qtyInputs)
        .filter(([_, q]) => Number(q) > 0)
        .map(([sku, q]) =>
          axios.post(`${API}/inventory/material-inward/sku`, {
            sku,
            quantity: Number(q),
          })
        )

      if (!reqs.length) {
        toast.error("Enter quantity")
        return
      }

      await Promise.all(reqs)
      toast.success("Stock added successfully")
      resetForm()
    } catch (e) {
      toast.error(e.response?.data?.detail || "Failed")
    } finally {
      setLoading(false)
    }
  }

  // ================= OUTWARD =================
  const submitOutward = async () => {
    if (!reason) {
      toast.error("Reason required")
      return
    }

    try {
      setLoading(true)

      const reqs = Object.entries(qtyInputs)
        .filter(([_, q]) => Number(q) > 0)
        .map(([sku, q]) =>
          axios.post(`${API}/inventory/material-outward/sku`, {
            sku,
            quantity: Number(q),
            reason,
          })
        )

      if (!reqs.length) {
        toast.error("Enter quantity")
        return
      }

      await Promise.all(reqs)
      toast.success("Stock deducted successfully")
      resetForm()
    } catch (e) {
      toast.error(e.response?.data?.detail || "Failed")
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setSkuInput("")
    setLookupData(null)
    setQtyInputs({})
    setReason("")
    skuRef.current?.focus()
  }

  // ================= TRANSACTIONS =================
  useEffect(() => {
    if (activeTab === "history") fetchTransactions()
  }, [activeTab, page, filterType])
const fetchTransactions = async () => {
  try {
    const params = { page, limit }
    if (filterType !== "all") params.type = filterType

    const res = await axios.get(`${API}/inventory/transactions`, { params })

    setTransactions(res.data.data || [])
    setTotal(res.data.total)
  } catch {
    toast.error("Failed to load transactions")
  }
}


  const filteredTransactions = useMemo(() => {
    if (!searchTerm) return transactions
    const q = searchTerm.toLowerCase()
    return transactions.filter(
      t =>
        t.product_name.toLowerCase().includes(q) ||
        t.product_code.toLowerCase().includes(q) ||
        t.variant_sku?.toLowerCase().includes(q)
    )
  }, [transactions, searchTerm])

  const totalPages = Math.ceil(total / limit)

  // ================= UI =================
  return (
    <div className="p-6 space-y-6">
      <h1 className="text-4xl font-bold">Inventory Management</h1>

      {/* TABS */}
      <div className="flex gap-2">
        <Button
          variant={activeTab === "inward" ? "default" : "outline"}
          onClick={() => setActiveTab("inward")}
        >
          <ArrowDownToLine className="w-4 h-4 mr-2" /> Inward
        </Button>

        <Button
          variant={activeTab === "outward" ? "default" : "outline"}
          onClick={() => setActiveTab("outward")}
        >
          <ArrowUpFromLine className="w-4 h-4 mr-2" /> Outward
        </Button>

        <Button
          variant={activeTab === "history" ? "default" : "outline"}
          onClick={() => setActiveTab("history")}
        >
          <History className="w-4 h-4 mr-2" /> History
        </Button>
      </div>

      {/* INWARD / OUTWARD */}
      {(activeTab === "inward" || activeTab === "outward") && (
        <Card className="max-w-2xl">
          <CardHeader>
            <CardTitle>
              {activeTab === "inward"
                ? "Material Inward"
                : "Material Outward"}
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <div>
              <Label>Scan / Enter SKU</Label>
              <Input
                ref={skuRef}
                value={skuInput}
                onChange={e => setSkuInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && lookupSku()}
                placeholder="Parent SKU or Variant SKU"
              />
              <Button className="mt-2" onClick={lookupSku}>
                Lookup
              </Button>
            </div>

            {lookupData && (
              <div className="bg-muted p-4 rounded space-y-3">
                <div className="font-semibold text-lg">
                  {lookupData.product_name}
                </div>

                <div className="text-sm">
                  Parent SKU:{" "}
                  <span className="font-mono">
                    {lookupData.parent_sku}
                  </span>
                </div>

                <div className="font-semibold">
                  Total Stock: {lookupData.total_stock}
                </div>

                {/* VARIANTS OR PRODUCT */}
                {(lookupData.variants?.length > 0
                  ? lookupData.variants
                  : [
                      {
                        v_sku: lookupData.parent_sku,
                        stock: lookupData.total_stock,
                      },
                    ]
                ).map(v => (
                  <div
                    key={v.v_sku}
                    className="bg-background border p-3 rounded space-y-2"
                  >
                    <div className="flex justify-between">
                      <span className="font-mono">{v.v_sku}</span>
                      <span className="font-semibold">
                        Stock: {v.stock}
                      </span>
                    </div>

                    <Input
                      type="number"
                      min="1"
                      placeholder="Qty"
                      value={qtyInputs[v.v_sku] || ""}
                      onChange={e =>
                        handleQtyChange(v.v_sku, e.target.value)
                      }
                    />
                  </div>
                ))}
              </div>
            )}

            {activeTab === "outward" && (
              <div>
                <Label>Reason</Label>
                <Input
                  value={reason}
                  onChange={e => setReason(e.target.value)}
                />
              </div>
            )}

            <Button
              className="w-full"
              disabled={loading}
              onClick={
                activeTab === "inward"
                  ? submitInward
                  : submitOutward
              }
            >
              Submit
            </Button>
          </CardContent>
        </Card>
      )}

      {/* HISTORY */}
      {activeTab === "history" && (
        <Card>
          <CardHeader>
            <CardTitle>Transaction History</CardTitle>
          </CardHeader>

          <CardContent className="space-y-4">
            <div className="flex gap-4">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
                <Input
                  className="pl-9"
                  placeholder="Search product / SKU"
                  value={searchTerm}
                  onChange={e => setSearchTerm(e.target.value)}
                />
              </div>

              <Select value={filterType} onValueChange={setFilterType}>
                <SelectTrigger className="w-32">
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
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b text-muted-foreground">
                    <th>Date</th>
                    <th>Type</th>
                    <th>Product</th>
                    <th>Code</th>
                    <th>V-SKU</th>
                    <th>Qty</th>
                    <th>Stock After</th>
                  </tr>
                </thead>

                <tbody>
                  {filteredTransactions.map(t => (
                    <tr key={t.id} className="border-b hover:bg-muted/40">
                      <td>{new Date(t.created_at).toLocaleString()}</td>
                      <td
                        className={
                          t.type === "IN"
                            ? "text-green-600 font-semibold"
                            : "text-red-600 font-semibold"
                        }
                      >
                        {t.type}
                      </td>
                      <td>{t.product_name}</td>
                      <td className="font-mono">{t.product_code}</td>
                      <td className="font-mono">{t.variant_sku || "—"}</td>
                      <td>{t.quantity}</td>
<td className="font-semibold">
  {t.variant_sku
    ? t.variant_stock_after ?? "—"
    : t.stock_after}
</td>

                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="flex justify-between items-center">
              <Button
                disabled={page === 1}
                onClick={() => setPage(p => p - 1)}
              >
                Prev
              </Button>

              <span>
                Page {page} / {totalPages || 1}
              </span>

              <Button
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
              >
                Next
              </Button>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
