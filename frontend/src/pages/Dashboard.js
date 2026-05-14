"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import axios from "axios"
import {
  AlertTriangle,
  ArrowDownToLine,
  ArrowUpFromLine,
  ChevronLeft,
  ChevronRight,
  DollarSign,
  Package,
  ShoppingCart,
  Users,
} from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { toast } from "sonner"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const getAuthHeaders = () => {
  const token = localStorage.getItem("token")
  return token ? { Authorization: `Bearer ${token}` } : {}
}

const currency = new Intl.NumberFormat("en-IN", {
  style: "currency",
  currency: "INR",
  maximumFractionDigits: 0,
})

const floatingCardClass =
  "rounded-2xl border border-white/10 bg-slate-950/35 backdrop-blur-md shadow-[0_18px_40px_rgba(15,23,42,0.18)]"

const floatingInnerRowClass =
  "rounded-xl border border-white/10 bg-white/[0.03] backdrop-blur-sm"

export default function Dashboard() {
  const [filter, setFilter] = useState("today")

  const [stats, setStats] = useState(null)
  const [salesData, setSalesData] = useState([])
  const [lowStock, setLowStock] = useState([])
  const [activity, setActivity] = useState([])
  const [inventoryMovement, setInventoryMovement] = useState([])
  const [topProducts, setTopProducts] = useState([])

  const [ageBucket, setAgeBucket] = useState("all")
  const [ageing, setAgeing] = useState([])
  const [agePage, setAgePage] = useState(1)
  const [agePages, setAgePages] = useState(1)

  const [lowStockPage, setLowStockPage] = useState(1)
  const lowStockLimit = 5

  const [loading, setLoading] = useState(true)

  const loadDashboard = useCallback(async () => {
    try {
      setLoading(true)
      const headers = getAuthHeaders()

      const res = await Promise.all([
        axios.get(`${API}/dashboard?filter=${filter}`, { headers }),
        axios.get(`${API}/dashboard/sales?filter=${filter}`, { headers }),
        axios.get(`${API}/dashboard/low-stock`, { headers }),
        axios.get(`${API}/dashboard/activity`, { headers }),
        axios.get(`${API}/dashboard/inventory-movement?days=7`, { headers }),
        axios.get(`${API}/dashboard/top-products?limit=6`, { headers }),
      ])

      setStats(res[0].data)
      setSalesData(res[1].data || [])
      setLowStock(res[2].data || [])
      setActivity(res[3].data || [])
      setInventoryMovement(res[4].data || [])
      setTopProducts(res[5].data || [])
    } catch {
      toast.error("Failed to load dashboard")
    } finally {
      setLoading(false)
    }
  }, [filter])

  const loadAgeing = useCallback(async () => {
    try {
      const bucketQuery = ageBucket === "all" ? "" : `&bucket=${ageBucket}`
      const res = await axios.get(
        `${API}/products/ageing?page=${agePage}&limit=10${bucketQuery}`,
        { headers: getAuthHeaders() }
      )

      setAgeing(res.data.data || [])
      setAgePages(res.data.total_pages || 1)
    } catch {
      toast.error("Failed to load stock ageing")
    }
  }, [ageBucket, agePage])

  useEffect(() => {
    loadDashboard()
  }, [loadDashboard])

  useEffect(() => {
    loadAgeing()
  }, [loadAgeing])

  const paginatedLowStock = useMemo(() => {
    const start = (lowStockPage - 1) * lowStockLimit
    return lowStock.slice(start, start + lowStockLimit)
  }, [lowStock, lowStockPage])

  const totalLowStockPages = Math.max(1, Math.ceil(lowStock.length / lowStockLimit))
  const inventoryTotals = useMemo(
    () =>
      inventoryMovement.reduce(
        (acc, item) => {
          acc.inward += Number(item.inward || 0)
          acc.outward += Number(item.outward || 0)
          return acc
        },
        { inward: 0, outward: 0 }
      ),
    [inventoryMovement]
  )

  if (loading || !stats) {
    return (
      <div className="flex h-full items-center justify-center text-slate-400">
        Loading dashboard...
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-6xl space-y-4 bg-[#050816] px-3 py-4 text-slate-100 md:px-6">
      <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-xl font-bold text-white md:text-2xl">Dashboard</h1>
          <p className="text-xs text-slate-400 md:text-sm">
            Business overview with live inventory insights
          </p>
        </div>

        <div className="w-full md:w-[180px]">
          <Select value={filter} onValueChange={setFilter}>
            <SelectTrigger className="h-10 rounded-xl border-slate-700 bg-[#0f172f] text-white">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="today">Today</SelectItem>
              <SelectItem value="yesterday">Yesterday</SelectItem>
              <SelectItem value="last_10_days">Last 10 Days</SelectItem>
              <SelectItem value="last_30_days">Last 30 Days</SelectItem>
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <StatCard title="Sales" value={currency.format(stats.total_sales || 0)} icon={DollarSign} />
        <StatCard title="Orders" value={stats.total_orders || 0} icon={ShoppingCart} />
        <StatCard title="Customers" value={stats.total_customers || 0} icon={Users} />
        <StatCard title="Low Stock" value={stats.low_stock_items || 0} icon={AlertTriangle} danger />
      </div>

      <div className="grid gap-3 lg:grid-cols-[1.2fr_0.8fr]">
        <ChartCard title="Sales Trend">
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={salesData}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 12,
                  color: "#fff",
                }}
              />
              <Line dataKey="total" stroke="#818cf8" strokeWidth={2.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <Card className={floatingCardClass}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-white">Inventory Summary</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 gap-2">
            <MiniKpi
              title="7D Inward"
              value={inventoryTotals.inward}
              icon={ArrowDownToLine}
              tone="emerald"
            />
            <MiniKpi
              title="7D Outward"
              value={inventoryTotals.outward}
              icon={ArrowUpFromLine}
              tone="rose"
            />
            <MiniKpi title="Low Alerts" value={lowStock.length} icon={AlertTriangle} tone="amber" />
            <MiniKpi title="Top Movers" value={topProducts.length} icon={Package} tone="indigo" />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-3 lg:grid-cols-2">
        <ChartCard title="Inventory Movement (7 Days)">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={inventoryMovement}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis dataKey="day" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 12,
                  color: "#fff",
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Bar dataKey="inward" fill="#10b981" radius={[6, 6, 0, 0]} />
              <Bar dataKey="outward" fill="#f43f5e" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Top Moving Products">
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={topProducts} layout="vertical" margin={{ left: 10, right: 10 }}>
              <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
              <XAxis type="number" tick={{ fontSize: 11, fill: "#94a3b8" }} />
              <YAxis
                dataKey="name"
                type="category"
                width={120}
                tick={{ fontSize: 11, fill: "#cbd5e1" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#0f172a",
                  border: "1px solid #334155",
                  borderRadius: 12,
                  color: "#fff",
                }}
              />
              <Bar dataKey="quantity" fill="#6366f1" radius={[0, 6, 6, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      <Card className={floatingCardClass}>
        <CardHeader className="flex flex-col gap-2 pb-2 md:flex-row md:items-center md:justify-between">
          <CardTitle className="text-sm font-semibold text-white">Stock Ageing</CardTitle>
          <Select
            value={ageBucket}
            onValueChange={value => {
              setAgeBucket(value)
              setAgePage(1)
            }}
          >
            <SelectTrigger className="h-9 w-full rounded-xl border-slate-700 bg-[#0f172f] text-xs text-white md:w-36">
              <SelectValue placeholder="Filter" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All</SelectItem>
              <SelectItem value="latest">Latest</SelectItem>
              <SelectItem value="new">New</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="old">Old</SelectItem>
              <SelectItem value="very_old">Very Old</SelectItem>
              <SelectItem value="dead_stock">Dead Stock</SelectItem>
            </SelectContent>
          </Select>
        </CardHeader>

        <CardContent className="space-y-2">
          {ageing.map(product => (
            <div
              key={product.sku}
              className={`flex items-center justify-between px-3 py-2 ${floatingInnerRowClass}`}
            >
              <div className="min-w-0">
                <p className="truncate text-xs font-medium text-white">{product.name}</p>
                <p className="text-[11px] text-slate-400">
                  {product.product_code} • {product.sku}
                </p>
              </div>

              <div className="text-right">
                <p className="text-xs font-semibold text-white">Qty {product.qty}</p>
                <Badge className="mt-1 bg-indigo-500/15 text-[10px] text-indigo-100">
                  {product.age_bucket.replace("_", " ").toUpperCase()}
                </Badge>
              </div>
            </div>
          ))}

          <Pager
            page={agePage}
            pages={agePages}
            onPrev={() => setAgePage(page => page - 1)}
            onNext={() => setAgePage(page => page + 1)}
          />
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Card className={floatingCardClass}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-white">Low Stock Alerts</CardTitle>
          </CardHeader>

          <CardContent className="space-y-2">
            {paginatedLowStock.map((product, index) => (
              <div
                key={`${product.product_name}-${index}`}
                className={`flex items-center justify-between px-3 py-2 ${floatingInnerRowClass}`}
              >
                <span className="text-xs text-slate-100">{product.product_name}</span>
                <Badge variant="destructive" className="text-[10px]">
                  {product.stock}/{product.min_stock}
                </Badge>
              </div>
            ))}

            <Pager
              page={lowStockPage}
              pages={totalLowStockPages}
              onPrev={() => setLowStockPage(page => page - 1)}
              onNext={() => setLowStockPage(page => page + 1)}
            />
          </CardContent>
        </Card>

        <Card className={floatingCardClass}>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm font-semibold text-white">Recent Activity</CardTitle>
          </CardHeader>

          <CardContent className="space-y-2">
            {activity.map((item, index) => (
              <div
                key={index}
                className={`px-3 py-2 text-xs text-slate-200 ${floatingInnerRowClass}`}
              >
                {item.text}
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

function StatCard({ title, value, icon: Icon, danger = false }) {
  return (
    <Card className="rounded-2xl border border-white/10 bg-[linear-gradient(135deg,rgba(15,23,42,0.72),rgba(15,23,42,0.42))] backdrop-blur-md shadow-[0_18px_36px_rgba(15,23,42,0.18)]">
      <CardContent className="flex items-center justify-between p-3">
        <div>
          <p className="text-[11px] uppercase tracking-[0.16em] text-slate-400">{title}</p>
          <p className={`mt-1 text-sm font-bold md:text-lg ${danger ? "text-rose-300" : "text-white"}`}>
            {value}
          </p>
        </div>
        <Icon className={`h-4 w-4 ${danger ? "text-rose-400" : "text-indigo-300"}`} />
      </CardContent>
    </Card>
  )
}

function ChartCard({ title, children }) {
  return (
    <Card className={floatingCardClass}>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-semibold text-white">{title}</CardTitle>
      </CardHeader>
      <CardContent>{children}</CardContent>
    </Card>
  )
}

function MiniKpi({ title, value, icon: Icon, tone = "indigo" }) {
  const toneMap = {
    emerald: "border-emerald-500/20 bg-emerald-500/10 text-emerald-200",
    rose: "border-rose-500/20 bg-rose-500/10 text-rose-200",
    amber: "border-amber-500/20 bg-amber-500/10 text-amber-200",
    indigo: "border-indigo-500/20 bg-indigo-500/10 text-indigo-100",
  }

  return (
    <div className={`rounded-2xl border p-3 ${toneMap[tone] || toneMap.indigo}`}>
      <div className="flex items-center justify-between">
        <p className="text-[11px] uppercase tracking-[0.16em]">{title}</p>
        <Icon className="h-4 w-4" />
      </div>
      <p className="mt-2 text-lg font-bold">{value}</p>
    </div>
  )
}

function Pager({ page, pages, onPrev, onNext }) {
  return (
    <div className="flex items-center justify-end gap-2 pt-1">
      <Button
        size="sm"
        variant="outline"
        className="border-slate-700 bg-[#0f172f] text-white hover:bg-[#16203d]"
        disabled={page === 1}
        onClick={onPrev}
      >
        <ChevronLeft className="h-4 w-4" />
      </Button>

      <span className="text-xs text-slate-400">
        {page} / {pages}
      </span>

      <Button
        size="sm"
        variant="outline"
        className="border-slate-700 bg-[#0f172f] text-white hover:bg-[#16203d]"
        disabled={page === pages}
        onClick={onNext}
      >
        <ChevronRight className="h-4 w-4" />
      </Button>
    </div>
  )
}
