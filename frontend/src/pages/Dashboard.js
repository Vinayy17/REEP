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
import { toast } from "sonner"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

/* ================= TOOLTIP ================= */
const SalesTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload

  return (
    <div className="bg-background border rounded-lg p-3 shadow-md text-sm">
      <p className="font-semibold mb-2">{label}</p>
      <p className="text-indigo-500">● Total: ₹{d.total.toFixed(2)}</p>
      <p className="text-green-600">● Paid: ₹{d.paid.toFixed(2)}</p>
      <p className="text-orange-500">● Pending: ₹{d.pending.toFixed(2)}</p>
      <p className="text-red-500">● Overdue: ₹{d.overdue.toFixed(2)}</p>
    </div>
  )
}

/* ================= DASHBOARD ================= */
export default function Dashboard() {
  const [filter, setFilter] = useState("today")

  const [stats, setStats] = useState(null)
  const [today, setToday] = useState(null)
  const [salesData, setSalesData] = useState([])
  const [hourlySales, setHourlySales] = useState([])
  const [topProducts, setTopProducts] = useState([])
  const [inventoryMove, setInventoryMove] = useState([])
  const [lowStock, setLowStock] = useState([])
  const [activity, setActivity] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboard()
  }, [filter])

  const loadDashboard = async () => {
    try {
      setLoading(true)

      const requests = [
        axios.get(`${API}/dashboard?filter=${filter}`),
        axios.get(`${API}/dashboard/sales?filter=${filter}`),
        axios.get(`${API}/dashboard/top-products`),
        axios.get(`${API}/dashboard/inventory-movement`),
        axios.get(`${API}/dashboard/low-stock`),
        axios.get(`${API}/dashboard/activity`),
      ]

      if (filter === "today") {
        requests.push(
          axios.get(`${API}/dashboard/today`),
          axios.get(`${API}/dashboard/hourly-sales`)
        )
      }

      const res = await Promise.all(requests)

      let i = 0
      setStats(res[i++].data)
      setSalesData(res[i++].data || [])
      setTopProducts(res[i++].data || [])
      setInventoryMove(res[i++].data || [])
      setLowStock(res[i++].data || [])
      setActivity(res[i++].data || [])

      if (filter === "today") {
        setToday(res[i++].data)
        setHourlySales(res[i++].data || [])
      } else {
        setToday(null)
        setHourlySales([])
      }
    } catch {
      toast.error("Failed to load dashboard")
    } finally {
      setLoading(false)
    }
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
          Business overview & performance
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

      {/* KPI CARDS */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard title="Total Sales" value={`₹${stats.total_sales.toFixed(2)}`} icon={DollarSign} />
        <StatCard title="Orders" value={stats.total_orders} icon={ShoppingCart} />
        <StatCard title="Customers" value={stats.total_customers} icon={Users} />
        <StatCard title="Low Stock" value={stats.low_stock_items} icon={AlertTriangle} />
      </div>

      {/* TODAY AT A GLANCE */}
      {filter === "today" && today && (
        <Card>
          <CardHeader>
            <CardTitle>Today at a Glance</CardTitle>
          </CardHeader>
          <CardContent className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
            <MiniStat label="Invoices" value={today.invoices_today} />
            <MiniStat label="Items Sold" value={today.items_sold_today} />
            <MiniStat label="Inventory OUT" value={today.inventory_out_today} />
            <MiniStat label="New Customers" value={today.new_customers_today} />
          </CardContent>
        </Card>
      )}

      {/* SALES TREND + SECONDARY GRAPH */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
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
                <Tooltip content={<SalesTooltip />} />
                <Line dataKey="total" stroke="#6366f1" strokeWidth={3} />
                <Line dataKey="paid" stroke="#22c55e" strokeWidth={2} />
                <Line dataKey="pending" stroke="#f97316" strokeDasharray="5 5" />
                <Line dataKey="overdue" stroke="#ef4444" strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* CONDITIONAL BAR GRAPH */}
        <Card>
          <CardHeader>
            <CardTitle>
              {filter === "today"
                ? "Hourly Sales (Today)"
                : filter === "last_10_days"
                ? "Last 10 Days Sales"
                : filter === "last_30_days"
                ? "Last 30 Days Sales"
                : "Sales Distribution"}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              {filter === "today" ? (
                <BarChart data={hourlySales}>
                  <XAxis dataKey="label" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="total" fill="#22c55e" radius={[6, 6, 0, 0]} />
                </BarChart>
              ) : (
                <BarChart data={salesData}>
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="total" fill="#6366f1" radius={[6, 6, 0, 0]} />
                </BarChart>
              )}
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* OTHER CHARTS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Top Selling Products</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={topProducts}>
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="quantity" fill="#6366f1" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Inventory Movement</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={inventoryMove}>
                <XAxis dataKey="day" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="inward" fill="#22c55e" />
                <Bar dataKey="outward" fill="#ef4444" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* PANELS */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <CardHeader>
            <CardTitle>Low Stock Alerts</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {lowStock.map((p, i) => (
              <div key={i} className="flex justify-between text-sm">
                <span>{p.product_name}</span>
                <span className="text-destructive font-semibold">
                  {p.stock}/{p.min_stock}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Recent Activity</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2 text-sm">
            {activity.map((a, i) => (
              <div key={i} className="flex items-center gap-2">
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

/* ================= COMPONENTS ================= */

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

const MiniStat = ({ label, value }) => (
  <div>
    <p className="text-xs text-muted-foreground">{label}</p>
    <p className="text-2xl font-bold">{value}</p>
  </div>
)
