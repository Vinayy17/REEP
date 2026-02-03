"use client"

import { useEffect, useState, useRef } from "react"
import axios from "axios"
import { motion, AnimatePresence } from "framer-motion"
import { Card, CardContent } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { toast } from "sonner"
import {
  Plus,
  Trash2,
  Search,
  CheckCircle,
  XCircle,
  ImageIcon,
  Camera,
} from "lucide-react"
import imageCompression from "browser-image-compression"


const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const priorities = ["urgent", "high", "normal", "low"]
const statuses = ["all", "pending", "completed", "rejected"]

const priorityStyle = {
  urgent: "bg-red-500/20 text-red-700 dark:text-red-400 border border-red-200 dark:border-red-800",
  high: "bg-orange-500/20 text-orange-700 dark:text-orange-400 border border-orange-200 dark:border-orange-800",
  normal: "bg-blue-500/20 text-blue-700 dark:text-blue-400 border border-blue-200 dark:border-blue-800",
  low: "bg-gray-500/20 text-gray-700 dark:text-gray-400 border border-gray-200 dark:border-gray-800",
}

const statusStyle = {
  pending: "bg-yellow-100 dark:bg-yellow-950 text-yellow-800 dark:text-yellow-200",
  completed: "bg-green-100 dark:bg-green-950 text-green-800 dark:text-green-200",
  rejected: "bg-red-100 dark:bg-red-950 text-red-800 dark:text-red-200",
}

export default function Requirements() {
  const user = typeof window !== "undefined" ? JSON.parse(localStorage.getItem("user") || "{}") : {}

  const [data, setData] = useState([])
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)

  const [search, setSearch] = useState("")
  const [status, setStatus] = useState("all")
  const [priority, setPriority] = useState("all")

  const [confirmCleanup, setConfirmCleanup] = useState(false)
  const [openCreate, setOpenCreate] = useState(false)
const cameraRefs = useRef([])

  const [form, setForm] = useState({
    customer_name: "",
    customer_phone: "",
    priority: "normal",
    items: [{ text: "", image_url: null }],
  })

  const fileInputRefs = useRef([])

  /* ================= FETCH ================= */

  useEffect(() => {
    fetchData()
  }, [page, status, priority, search])
const compressImage = async (file) => {
  const options = {
    maxSizeMB: 0.5,           // ✅ 500 KB target
    maxWidthOrHeight: 1280,   // ✅ good for camera photos
    useWebWorker: true,
    initialQuality: 0.7,
  }

  try {
    const compressedFile = await imageCompression(file, options)
    return compressedFile
  } catch (err) {
    console.error("Compression failed", err)
    return file // fallback
  }
}
  const fetchData = async () => {
    try {
      const token = localStorage.getItem("token")
      const res = await axios.get(`${API}/requirements`, {
        headers: { Authorization: `Bearer ${token}` },
        params: { page, limit: 10, status, priority, search },
      })

      setData(res.data.data)
      setTotalPages(res.data.total_pages)
    } catch {
      toast.error("Failed to load requirements")
    }
  }

  /* ================= CREATE ================= */

  const addItem = () =>
    setForm(f => ({
      ...f,
      items: [...f.items, { text: "", image_url: null }],
    }))

  const updateItem = (i, key, value) => {
    const items = [...form.items]
    items[i] = { ...items[i], [key]: value }
    setForm({ ...form, items })
  }

  const removeItem = (i) =>
    setForm(f => ({
      ...f,
      items: f.items.filter((_, idx) => idx !== i),
    }))

  const uploadImage = async (file, index) => {
    try {
      const token = localStorage.getItem("token")
      const fd = new FormData()
      fd.append("file", file)

      const res = await axios.post(`${API}/upload/requirement-image`, fd, {
        headers: { Authorization: `Bearer ${token}` },
      })

      updateItem(index, "image_url", res.data.url)
      toast.success("Image uploaded")
    } catch {
      toast.error("Upload failed")
    }
  }

  const submitRequirement = async () => {
    try {
      const token = localStorage.getItem("token")

      await axios.post(
        `${API}/requirements`,
        {
          customer_name: form.customer_name,
          customer_phone: form.customer_phone,
          priority: form.priority,
          requirement_items: form.items.filter(i => i.text),
        },
        { headers: { Authorization: `Bearer ${token}` } }
      )

      toast.success("Requirement created")
      setOpenCreate(false)
      setForm({
        customer_name: "",
        customer_phone: "",
        priority: "normal",
        items: [{ text: "", image_url: null }],
      })
      fetchData()
    } catch (e) {
      toast.error(e.response?.data?.detail || "Create failed")
    }
  }

  /* ================= CLEANUP ================= */

  const cleanupRequirements = async () => {
    try {
      const token = localStorage.getItem("token")
      await axios.delete(`${API}/requirements/cleanup`, {
        headers: { Authorization: `Bearer ${token}` },
      })

      toast.success("Old requirements cleaned")
      setConfirmCleanup(false)
      setPage(1)
      fetchData()
    } catch {
      toast.error("Cleanup failed (Admin only)")
    }
  }

  /* ================= UI ================= */

  return (
    <div className="min-h-screen bg-gradient-to-br from-background to-muted/30">
      <div className="p-4 md:p-8 max-w-7xl mx-auto space-y-6">

        {/* HEADER */}
        <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-accent">
              Requirements
            </h1>
            <p className="text-sm md:text-base text-muted-foreground mt-2">
              Track & convert customer needs into solutions
            </p>
          </div>

          <div className="flex gap-2 flex-wrap">
            <Button onClick={() => setOpenCreate(true)} className="gap-2">
              <Plus className="w-4 h-4" /> New Requirement
            </Button>

            {user?.role === "admin" && (
              <Button variant="destructive" onClick={() => setConfirmCleanup(true)} className="gap-2">
                <Trash2 className="w-4 h-4" /> Cleanup
              </Button>
            )}
          </div>
        </div>

        {/* SEARCH + FILTERS */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="relative md:col-span-2">
            <Search className="absolute left-3 top-3 w-4 h-4 text-muted-foreground" />
            <Input
              className="pl-9 h-10"
              placeholder="Search by customer name, phone, or user..."
              value={search}
              onChange={e => setSearch(e.target.value)}
            />
          </div>

          <select
            value={status}
            onChange={e => setStatus(e.target.value)}
            className="h-10 px-3 rounded-lg border border-border bg-card text-foreground cursor-pointer hover:border-primary/50 transition-colors"
          >
            {statuses.map(s => (
              <option key={s} value={s} className="bg-card text-foreground">
                {s.charAt(0).toUpperCase() + s.slice(1)}
              </option>
            ))}
          </select>
        </div>

        {/* PRIORITY FILTER */}
        <div className="flex gap-2 overflow-x-auto pb-2">
          <Button
            size="sm"
            variant={priority === "all" ? "default" : "outline"}
            onClick={() => setPriority("all")}
            className="whitespace-nowrap"
          >
            All
          </Button>

          {priorities.map(p => (
            <Button
              key={p}
              size="sm"
              className={`whitespace-nowrap ${priorityStyle[p]}`}
              variant={priority === p ? "default" : "outline"}
              onClick={() => setPriority(p)}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </Button>
          ))}
        </div>

        {/* LIST - Cards on mobile, Table on desktop */}
        <AnimatePresence>
          {/* Mobile View - Cards */}
          <div className="md:hidden space-y-4">
            {data.map((r, idx) => (
              <motion.div
                key={r.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ delay: idx * 0.05 }}
              >
                <Card className="hover:shadow-lg transition-shadow border-l-4 border-l-primary">
                  <CardContent className="p-4 space-y-3">
                    {/* Header */}
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-semibold text-lg text-foreground">{r.customer_name}</div>
                        <div className="text-xs text-muted-foreground">{r.customer_phone}</div>
                      </div>
                      <span className={`px-3 py-1 text-xs font-medium rounded-full ${priorityStyle[r.priority]}`}>
                        {r.priority.toUpperCase()}
                      </span>
                    </div>

                    {/* Status Badge */}
                    <div className="flex gap-2">
                      <span className={`px-2 py-1 text-xs font-medium rounded-md ${statusStyle[r.status] || 'bg-gray-100 dark:bg-gray-900'}`}>
                        {r.status.charAt(0).toUpperCase() + r.status.slice(1)}
                      </span>
                    </div>

                    {/* Items */}
                    <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                      <div className="text-xs font-semibold text-muted-foreground uppercase">Items</div>
                      <ul className="text-sm space-y-2">
                        {Array.isArray(r.requirement_items) &&
                          r.requirement_items.map((it, i) => (
                            <li key={i} className="flex items-center gap-2 text-foreground">
                              <div className="w-2 h-2 rounded-full bg-primary" />
                              <span>{it.text}</span>
                              {it.image_url && (
                                <a href={it.image_url} target="_blank" rel="noopener noreferrer" className="ml-auto">
                                  <ImageIcon className="w-4 h-4 text-primary hover:text-primary/80 transition-colors" />
                                </a>
                              )}
                            </li>
                          ))}
                      </ul>
                    </div>

                    {/* Footer */}
                    <div className="text-xs text-muted-foreground space-y-1 border-t pt-3">
                      <div>By <span className="font-medium text-foreground">{r.created_by}</span></div>
                      <div>{new Date(r.created_at).toLocaleString("en-IN")}</div>
                    </div>

                    {/* Actions */}
                    {r.status === "pending" && (
                      <div className="flex gap-2 pt-2">
                        <Button
                          size="sm"
                          variant="outline"
                          className="flex-1 gap-2 bg-transparent"
                          onClick={() =>
                            axios.post(`${API}/requirements/${r.id}/complete`, {}, {
                              headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
                            }).then(fetchData)
                          }>
                          <CheckCircle className="w-4 h-4" /> Complete
                        </Button>

                        <Button
                          size="sm"
                          variant="destructive"
                          className="flex-1 gap-2"
                          onClick={() =>
                            axios.post(`${API}/requirements/${r.id}/reject`, {}, {
                              headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
                            }).then(fetchData)
                          }>
                          <XCircle className="w-4 h-4" /> Reject
                        </Button>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>

          {/* Desktop View - Table */}
          <div className="hidden md:block overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-border">
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Customer</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Phone</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Items</th>
                  <th className="text-center py-3 px-4 font-semibold text-foreground">Priority</th>
                  <th className="text-center py-3 px-4 font-semibold text-foreground">Status</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Created By</th>
                  <th className="text-left py-3 px-4 font-semibold text-foreground">Date</th>
                  <th className="text-center py-3 px-4 font-semibold text-foreground">Actions</th>
                </tr>
              </thead>
              <tbody>
                {data.map((r, idx) => (
                  <motion.tr
                    key={r.id}
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: idx * 0.05 }}
                    className="border-b border-border hover:bg-muted/50 transition-colors"
                  >
                    <td className="py-3 px-4 text-foreground font-medium">{r.customer_name}</td>
                    <td className="py-3 px-4 text-foreground text-sm">{r.customer_phone}</td>
                    <td className="py-3 px-4">
                      <div className="text-sm text-foreground max-w-xs">
                        {Array.isArray(r.requirement_items) &&
                          r.requirement_items.map((it, i) => (
                            <div key={i} className="flex items-center gap-1">
                              <span>{it.text}</span>
                              {it.image_url && (
                                <a href={it.image_url} target="_blank" rel="noopener noreferrer">
                                  <ImageIcon className="w-4 h-4 text-primary hover:text-primary/80 transition-colors" />
                                </a>
                              )}
                            </div>
                          ))}
                      </div>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-3 py-1 text-xs font-medium rounded-full ${priorityStyle[r.priority]}`}>
                        {r.priority.toUpperCase()}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`px-2 py-1 text-xs font-medium rounded-md inline-block ${statusStyle[r.status] || 'bg-gray-100 dark:bg-gray-900'}`}>
                        {r.status.charAt(0).toUpperCase() + r.status.slice(1)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-foreground text-sm">{r.created_by}</td>
                    <td className="py-3 px-4 text-foreground text-sm">{new Date(r.created_at).toLocaleString("en-IN")}</td>
                    <td className="py-3 px-4 text-center">
                      {r.status === "pending" && (
                        <div className="flex gap-1 justify-center">
                          <Button
                            size="sm"
                            variant="outline"
                            className="h-8 w-8 p-0 bg-transparent"
                            title="Complete"
                            onClick={() =>
                              axios.post(`${API}/requirements/${r.id}/complete`, {}, {
                                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
                              }).then(fetchData)
                            }>
                            <CheckCircle className="w-4 h-4" />
                          </Button>

                          <Button
                            size="sm"
                            variant="destructive"
                            className="h-8 w-8 p-0"
                            title="Reject"
                            onClick={() =>
                              axios.post(`${API}/requirements/${r.id}/reject`, {}, {
                                headers: { Authorization: `Bearer ${localStorage.getItem("token")}` },
                              }).then(fetchData)
                            }>
                            <XCircle className="w-4 h-4" />
                          </Button>
                        </div>
                      )}
                    </td>
                  </motion.tr>
                ))}
              </tbody>
            </table>
          </div>
        </AnimatePresence>

        {/* PAGINATION */}
        <div className="flex justify-center items-center gap-4 pt-4">
          <Button
            disabled={page === 1}
            onClick={() => setPage(p => p - 1)}
            variant={page === 1 ? "outline" : "default"}
          >
            ← Previous
          </Button>
          <span className="text-sm font-medium text-foreground">
            Page <span className="text-primary font-bold">{page}</span> of <span className="text-primary font-bold">{totalPages}</span>
          </span>
          <Button
            disabled={page === totalPages}
            onClick={() => setPage(p => p + 1)}
            variant={page === totalPages ? "outline" : "default"}
          >
            Next →
          </Button>
        </div>

        {/* CREATE MODAL */}
        <Dialog open={openCreate} onOpenChange={setOpenCreate}>
          <DialogContent className="max-w-2xl">
            <DialogHeader>
              <DialogTitle className="text-2xl">Create New Requirement</DialogTitle>
            </DialogHeader>

            <div className="space-y-4 max-h-[70vh] overflow-y-auto">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  placeholder="Customer name"
                  className="h-10"
                  value={form.customer_name}
                  onChange={e => setForm({ ...form, customer_name: e.target.value })} />

                <Input
                  placeholder="Customer phone"
                  className="h-10"
                  value={form.customer_phone}
                  onChange={e => setForm({ ...form, customer_phone: e.target.value })} />
              </div>

              <div>
                <label className="text-sm font-medium text-foreground block mb-2">Priority</label>
                <select
                  value={form.priority}
                  onChange={e => setForm({ ...form, priority: e.target.value })}
                  className="w-full h-10 px-3 rounded-lg border border-border bg-card text-foreground cursor-pointer hover:border-primary/50 transition-colors"
                >
                  {priorities.map(p => (
                    <option key={p} value={p} className="bg-card text-foreground">
                      {p.charAt(0).toUpperCase() + p.slice(1)}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <div className="flex items-center justify-between mb-3">
                  <label className="text-sm font-medium text-foreground">Items</label>
                  <Button size="sm" variant="outline" onClick={addItem} className="gap-1 bg-transparent">
                    <Plus className="w-3 h-3" /> Add Item
                  </Button>
                </div>

                <div className="space-y-3">
                  {form.items.map((item, i) => (
                    <div key={i} className="flex gap-2 items-end">
                      <div className="flex-1 space-y-1">
                        <Input
                          placeholder={`Item ${i + 1}`}
                          className="h-9"
                          value={item.text}
                          onChange={e => updateItem(i, "text", e.target.value)} />
                      </div>

                      <Button
                        size="sm"
                        variant="ghost"
                        className="h-9 w-9 p-0"
    onClick={() => cameraRefs.current[i]?.click()}
                        title="Upload image"
                      >
                        <Camera className="w-4 h-4" />
                      </Button>

                     <input
  ref={el => (cameraRefs.current[i] = el)}
  type="file"
  accept="image/*"
  capture="environment"
  className="hidden"
  onChange={async e => {
    const file = e.target.files?.[0]
    if (!file) return

    const compressed = await compressImage(file)
    uploadImage(compressed, i)

    e.target.value = "" // reset input
  }}
/>


                      {item.image_url && (
                        <a href={item.image_url} target="_blank" rel="noopener noreferrer" className="h-9 w-9 flex items-center justify-center">
                          <ImageIcon className="w-4 h-4 text-primary hover:text-primary/80 transition-colors" />
                        </a>
                      )}

                      {form.items.length > 1 && (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="h-9 w-9 p-0 text-destructive hover:text-destructive"
                          onClick={() => removeItem(i)}
                        >
                          <Trash2 className="w-4 h-4" />
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex gap-2 justify-end pt-4">
                <Button variant="outline" onClick={() => setOpenCreate(false)}>
                  Cancel
                </Button>
                <Button
                  onClick={submitRequirement}
                  disabled={!form.customer_name || !form.customer_phone || !form.items.some(i => i.text)}
                  className="gap-2"
                >
                  <Plus className="w-4 h-4" /> Create Requirement
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>

        {/* CLEANUP CONFIRMATION MODAL */}
        <Dialog open={confirmCleanup} onOpenChange={setConfirmCleanup}>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Cleanup Old Requirements?</DialogTitle>
            </DialogHeader>
            <p className="text-sm text-muted-foreground py-4">
              This will delete all requirements older than 30 days. This action cannot be undone.
            </p>
            <div className="flex gap-2 justify-end">
              <Button variant="outline" onClick={() => setConfirmCleanup(false)}>
                Cancel
              </Button>
              <Button variant="destructive" onClick={cleanupRequirements}>
                Cleanup
              </Button>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}
