"use client"

import { useEffect, useState } from "react"
import axios from "axios"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  DollarSign,
  ShoppingCart,
  Users,
  AlertTriangle,
  TrendingUp,
  Activity,
  ChevronLeft,
  ChevronRight,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { toast } from "sonner"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

/* ================= DASHBOARD ================= */
export default function Dashboard() {
  const [filter, setFilter] = useState("today")

  const [stats, setStats] = useState(null)
  const [salesData, setSalesData] = useState([])
  const [topProducts, setTopProducts] = useState([])
  const [inventoryMove, setInventoryMove] = useState([])
  const [lowStock, setLowStock] = useState([])
  const [activity, setActivity] = useState([])

  // 🔥 Stock Ageing
  const [ageBucket, setAgeBucket] = useState("all")
  const [ageing, setAgeing] = useState([])
  const [agePage, setAgePage] = useState(1)
  const [agePages, setAgePages] = useState(1)

  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboard()
  }, [filter])

  useEffect(() => {
    loadAgeing()
  }, [ageBucket, agePage])

  const loadDashboard = async () => {
    try {
      setLoading(true)
      const res = await Promise.all([
        axios.get(`${API}/dashboard?filter=${filter}`),
        axios.get(`${API}/dashboard/sales?filter=${filter}`),
        axios.get(`${API}/dashboard/top-products`),
        axios.get(`${API}/dashboard/inventory-movement`),
        axios.get(`${API}/dashboard/low-stock`),
        axios.get(`${API}/dashboard/activity`),
      ])

      setStats(res[0].data)
      setSalesData(res[1].data || [])
      setTopProducts(res[2].data || [])
      setInventoryMove(res[3].data || [])
      setLowStock(res[4].data || [])
      setActivity(res[5].data || [])
    } catch {
      toast.error("Failed to load dashboard")
    } finally {
      setLoading(false)
    }
  }

  const loadAgeing = async () => {
    const bucketQuery = ageBucket === "all" ? "" : `&bucket=${ageBucket}`
    const res = await axios.get(
      `${API}/products/ageing?page=${agePage}&limit=10${bucketQuery}`
    )
    setAgeing(res.data.data)
    setAgePages(res.data.total_pages)
  }

  if (loading) {
    return (
      <div className="flex h-full items-center justify-center text-muted-foreground">
        Loading dashboard...
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      {/* HEADER */}
      <div>
        <h1 className="text-4xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Business overview & stock insights
        </p>
      </div>

      {/* FILTER */}
      <div className="flex justify-end">
        <select
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="border rounded-md px-3 py-1 text-sm bg-background"
        >
          <option value="today">Today</option>
          <option value="yesterday">Yesterday</option>
          <option value="last_10_days">Last 10 Days</option>
          <option value="last_30_days">Last 30 Days</option>
        </select>
      </div>

      {/* KPI */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Sales" value={`₹${stats.total_sales.toFixed(2)}`} icon={DollarSign} />
        <StatCard title="Orders" value={stats.total_orders} icon={ShoppingCart} />
        <StatCard title="Customers" value={stats.total_customers} icon={Users} />
        <StatCard title="Low Stock" value={stats.low_stock_items} icon={AlertTriangle} />
      </div>

      {/* SALES TREND */}
      <Card>
        <CardHeader>
          <CardTitle>Sales Trend</CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={salesData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Line dataKey="total" stroke="#6366f1" strokeWidth={3} />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* 🔥 STOCK AGEING */}
      <Card>
        <CardHeader className="flex flex-row justify-between items-center">
          <CardTitle>Stock Ageing</CardTitle>

        <Select
  value={ageBucket}
  onValueChange={(v) => {
    setAgeBucket(v)
    setAgePage(1)
  }}
>
  <SelectTrigger className="w-40 bg-black text-white border border-border focus:ring-1 focus:ring-primary">
    <SelectValue placeholder="Filter" />
  </SelectTrigger>

  <SelectContent className="bg-black text-white border border-border">
    <SelectItem value="all" className="focus:bg-white/10">
      All
    </SelectItem>
    <SelectItem value="latest" className="focus:bg-white/10">
      Latest
    </SelectItem>
    <SelectItem value="new" className="focus:bg-white/10">
      New
    </SelectItem>
    <SelectItem value="medium" className="focus:bg-white/10">
      Medium
    </SelectItem>
    <SelectItem value="old" className="focus:bg-white/10">
      Old
    </SelectItem>
    <SelectItem value="very_old" className="focus:bg-white/10">
      Very Old
    </SelectItem>
    <SelectItem value="dead_stock" className="focus:bg-white/10">
      Dead Stock
    </SelectItem>
  </SelectContent>
</Select>

        </CardHeader>

        <CardContent className="space-y-3">
          {ageing.map(p => (
            <div key={p.sku} className="flex justify-between border rounded-lg p-3">
              <div>
                <p className="font-medium">{p.name}</p>
                <p className="text-xs text-muted-foreground">
                  {p.product_code} • {p.sku}
                </p>
                {p.first_inward_date && (
                  <p className="text-xs text-muted-foreground">
                    Inward: {new Date(p.first_inward_date).toLocaleDateString()}
                  </p>
                )}
              </div>

              <div className="text-right space-y-1">
                <p className="font-semibold">Qty: {p.qty}</p>
                <Badge>{p.age_bucket.replace("_", " ").toUpperCase()}</Badge>
              </div>
            </div>
          ))}

          {/* PAGINATION */}
          <div className="flex justify-end gap-2 pt-2">
            <Button
              size="sm"
              variant="outline"
              disabled={agePage === 1}
              onClick={() => setAgePage(p => p - 1)}
            >
              <ChevronLeft className="w-4 h-4" />
            </Button>
            <Button
              size="sm"
              variant="outline"
              disabled={agePage === agePages}
              onClick={() => setAgePage(p => p + 1)}
            >
              <ChevronRight className="w-4 h-4" />
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* LOW STOCK + ACTIVITY */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Low Stock Alerts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {lowStock.map((p, i) => (
              <div key={i} className="flex justify-between bg-muted rounded-md p-2">
                <span>{p.product_name}</span>
                <Badge variant="destructive">
                  {p.stock}/{p.min_stock}
                </Badge>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {activity.map((a, i) => (
              <div key={i} className="flex gap-2 text-sm items-center">
                <Activity className="w-4 h-4 text-muted-foreground" />
                <span>{a.text}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

/* ================= SMALL COMPONENTS ================= */

const StatCard = ({ title, value, icon: Icon }) => (
  <Card>
    <CardHeader className="flex justify-between items-center pb-2">
      <CardTitle className="text-sm text-muted-foreground">{title}</CardTitle>
      <Icon className="w-5 h-5 text-primary" />
    </CardHeader>
    <CardContent>
      <div className="text-3xl font-bold">{value}</div>
      <p className="text-xs text-muted-foreground flex items-center gap-1 mt-1">
        <TrendingUp className="w-3 h-3" /> Updated
      </p>
    </CardContent>
  </Card>
)
