"use client"

import { useEffect, useMemo, useState } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { Plus, Trash2, Pencil, Package2, Search, Layers3, Tag, Boxes } from "lucide-react"
import { toast } from "sonner"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const formatCurrency = (amount) => {
  const value = Number(amount || 0)
  return Number.isNaN(value) ? "0.00" : value.toFixed(2)
}

const Combo = () => {
  const [products, setProducts] = useState([])
  const [categories, setCategories] = useState([])
  const [combos, setCombos] = useState([])

  const [editingComboId, setEditingComboId] = useState(null)
  const [comboName, setComboName] = useState("")
  const [comboPrice, setComboPrice] = useState("")
  const [selectedItems, setSelectedItems] = useState([])

  const [search, setSearch] = useState("")
  const [debouncedSearch, setDebouncedSearch] = useState("")
  const [categoryFilter, setCategoryFilter] = useState("all")

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 350)
    return () => clearTimeout(timer)
  }, [search])

  const fetchProducts = async () => {
    const res = await axios.get(`${API}/products`, {
      params: {
        search: debouncedSearch || undefined,
        category_id: categoryFilter !== "all" ? categoryFilter : undefined,
        limit: 50
      }
    })
    setProducts(res.data.data || res.data || [])
  }

  const fetchCategories = async () => {
    const res = await axios.get(`${API}/categories`)
    setCategories(res.data || [])
  }

  const fetchCombos = async () => {
    const res = await axios.get(`${API}/combos`)
    setCombos(res.data || [])
  }

  useEffect(() => {
    fetchCategories()
    fetchCombos()
  }, [])

  useEffect(() => {
    fetchProducts()
  }, [debouncedSearch, categoryFilter])

  const addToCombo = (product) => {
    const existing = selectedItems.find((item) => item.product_id === product.id)

    if (existing) {
      setSelectedItems((prev) =>
        prev.map((item) =>
          item.product_id === product.id
            ? { ...item, quantity: item.quantity + 1 }
            : item
        )
      )
      return
    }

    setSelectedItems((prev) => [
      ...prev,
      {
        product_id: product.id,
        name: product.name,
        sku: product.sku,
        quantity: 1
      }
    ])
  }

  const updateQty = (id, qty) => {
    const parsedQty = Math.max(1, Number(qty) || 1)
    setSelectedItems((prev) =>
      prev.map((item) =>
        item.product_id === id
          ? { ...item, quantity: parsedQty }
          : item
      )
    )
  }

  const removeItem = (id) => {
    setSelectedItems((prev) => prev.filter((item) => item.product_id !== id))
  }

  const saveCombo = async () => {
    if (!comboName.trim() || selectedItems.length === 0) {
      toast.error("Enter combo name and add products")
      return
    }

    const payload = {
      name: comboName.trim(),
      price: Number(comboPrice || 0),
      items: selectedItems.map((item) => ({
        product_id: item.product_id,
        quantity: Number(item.quantity)
      }))
    }

    try {
      if (editingComboId) {
        await axios.put(`${API}/combos/${editingComboId}`, payload)
        toast.success("Combo updated")
      } else {
        await axios.post(`${API}/combos`, payload)
        toast.success("Combo created")
      }

      resetForm()
      fetchCombos()
    } catch {
      toast.error("Failed to save combo")
    }
  }

  const deleteCombo = async (id) => {
    if (!window.confirm("Delete this combo?")) return

    try {
      await axios.delete(`${API}/combos/${id}`)
      toast.success("Combo deleted")
      fetchCombos()
    } catch {
      toast.error("Failed to delete combo")
    }
  }

  const handleEdit = (combo) => {
    setEditingComboId(combo.id)
    setComboName(combo.name || "")
    setComboPrice(String(combo.price || ""))
    setSelectedItems(
      (combo.items || []).map((item) => ({
        product_id: item.product_id,
        name: item.product_name,
        sku: item.sku,
        quantity: item.quantity
      }))
    )
  }

  const resetForm = () => {
    setEditingComboId(null)
    setComboName("")
    setComboPrice("")
    setSelectedItems([])
  }

  const selectedSummary = useMemo(
    () => selectedItems.reduce((sum, item) => sum + Number(item.quantity || 0), 0),
    [selectedItems]
  )

  return (
    <div className="h-screen overflow-y-auto bg-[radial-gradient(circle_at_top_left,#1f4b99_0%,#0f172a_34%,#09111f_100%)] px-4 py-5 text-white md:px-6">
      <div className="mx-auto flex min-h-full max-w-7xl flex-col gap-6 xl:flex-row">
        <section className="xl:sticky xl:top-5 xl:h-[calc(100vh-2.5rem)] xl:w-[430px] xl:flex-shrink-0">
          <Card className="flex h-full flex-col overflow-hidden rounded-[30px] border border-white/10 bg-[#0f172a]/85 shadow-[0_30px_90px_rgba(0,0,0,0.35)] backdrop-blur">
            <div className="border-b border-white/10 bg-[linear-gradient(135deg,#15316b_0%,#1f4d95_48%,#2f7ac2_100%)] p-6">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <p className="text-xs uppercase tracking-[0.32em] text-white/65">Combo Studio</p>
                  <h2 className="mt-2 text-2xl font-semibold">
                    {editingComboId ? "Edit Bundle" : "Create Bundle"}
                  </h2>
                  <p className="mt-2 max-w-sm text-sm text-white/80">
                    Build product packs in one place with dark invoice-style visuals and clean horizontal previews.
                  </p>
                </div>
                <div className="rounded-2xl bg-white/12 p-3">
                  <Layers3 className="h-6 w-6" />
                </div>
              </div>

              <div className="mt-5 grid grid-cols-2 gap-3">
                <div className="rounded-2xl bg-white/12 p-3">
                  <p className="text-xs text-white/65">Lines</p>
                  <p className="mt-1 text-2xl font-semibold">{selectedItems.length}</p>
                </div>
                <div className="rounded-2xl bg-white/12 p-3">
                  <p className="text-xs text-white/65">Units</p>
                  <p className="mt-1 text-2xl font-semibold">{selectedSummary}</p>
                </div>
              </div>
            </div>

            <CardContent className="flex-1 space-y-5 overflow-y-auto p-5">
              <div className="grid gap-3">
                <Input
                  value={comboName}
                  onChange={(e) => setComboName(e.target.value)}
                  placeholder="Combo name"
                  className="h-12 rounded-2xl border-white/10 bg-white/5 text-white placeholder:text-slate-400"
                />
                <Input
                  value={comboPrice}
                  onChange={(e) => setComboPrice(e.target.value)}
                  placeholder="Combo price"
                  type="number"
                  className="h-12 rounded-2xl border-white/10 bg-white/5 text-white placeholder:text-slate-400"
                />
              </div>

              <div className="rounded-[26px] border border-white/10 bg-white/5 p-4">
                <div className="mb-3 flex items-center justify-between">
                  <div>
                    <p className="text-sm font-semibold text-white">Selected Products</p>
                    <p className="text-xs text-slate-300">Adjust quantity and save</p>
                  </div>
                  <div className="rounded-full bg-white/10 px-3 py-1 text-xs font-medium text-slate-200">
                    {selectedSummary} qty
                  </div>
                </div>

                <div className="space-y-3">
                  {selectedItems.length === 0 && (
                    <div className="rounded-2xl border border-dashed border-white/15 bg-black/10 px-4 py-8 text-center text-sm text-slate-300">
                      Add products from the right panel
                    </div>
                  )}

                  {selectedItems.map((item) => (
                    <div
                      key={item.product_id}
                      className="flex items-center gap-3 rounded-2xl border border-white/10 bg-[#111b2f] p-3"
                    >
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-white/10 text-sky-300">
                        <Package2 className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold text-white">{item.name}</p>
                        <p className="truncate text-xs text-slate-400">{item.sku || "No SKU"}</p>
                      </div>
                      <Input
                        type="number"
                        min="1"
                        value={item.quantity}
                        onChange={(e) => updateQty(item.product_id, e.target.value)}
                        className="h-10 w-20 rounded-xl border-white/10 bg-black/10 text-center text-white"
                      />
                      <Button
                        size="icon"
                        variant="destructive"
                        onClick={() => removeItem(item.product_id)}
                        className="h-10 w-10 rounded-xl"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  ))}
                </div>
              </div>

              <div className="flex gap-3">
                <Button onClick={saveCombo} className="h-12 flex-1 rounded-2xl bg-sky-500 text-slate-950 hover:bg-sky-400">
                  {editingComboId ? "Update Combo" : "Create Combo"}
                </Button>
                {editingComboId && (
                  <Button variant="outline" onClick={resetForm} className="h-12 rounded-2xl border-white/15 bg-white/5 text-white hover:bg-white/10">
                    Cancel
                  </Button>
                )}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="min-w-0 flex-1 space-y-6">
          <Card className="rounded-[30px] border border-white/10 bg-[#0f172a]/80 shadow-[0_30px_90px_rgba(0,0,0,0.28)] backdrop-blur">
            <CardContent className="p-5 md:p-6">
              <div className="mb-5 flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Product Picker</p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">Browse Products</h3>
                  <p className="mt-1 text-sm text-slate-300">
                    Search, filter, and add products into your combo builder.
                  </p>
                </div>
                <div className="rounded-full bg-white/10 px-4 py-2 text-sm font-medium text-slate-200">
                  {products.length} products
                </div>
              </div>

              <div className="mb-5 grid gap-3 lg:grid-cols-[1fr_240px]">
                <div className="relative">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
                  <Input
                    placeholder="Search products"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="h-12 rounded-2xl border-white/10 bg-white/5 pl-11 text-white placeholder:text-slate-400"
                  />
                </div>

                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="h-12 rounded-2xl border-white/10 bg-white/5 text-white">
                    <SelectValue placeholder="Category" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Categories</SelectItem>
                    {categories.map((category) => (
                      <SelectItem key={category.id} value={category.id}>
                        {category.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                {products.map((product) => (
                  <button
                    key={product.id}
                    type="button"
                    onClick={() => addToCombo(product)}
                    className="group flex min-h-[122px] items-center gap-4 rounded-[28px] border border-white/10 bg-[linear-gradient(135deg,rgba(255,255,255,0.07)_0%,rgba(255,255,255,0.03)_100%)] p-4 text-left transition-all duration-200 hover:-translate-y-0.5 hover:border-sky-400/60 hover:bg-[linear-gradient(135deg,rgba(56,189,248,0.18)_0%,rgba(255,255,255,0.05)_100%)] hover:shadow-[0_22px_50px_rgba(14,165,233,0.15)]"
                  >
                    <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-2xl bg-sky-500/15 text-sky-300">
                      <Package2 className="h-6 w-6" />
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-white">{product.name}</p>
                      <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-300">
                        <span className="inline-flex items-center gap-1 rounded-full bg-white/10 px-2.5 py-1">
                          <Tag className="h-3 w-3" />
                          {product.sku || "No SKU"}
                        </span>
                        {product.category_name && (
                          <span className="rounded-full bg-amber-500/15 px-2.5 py-1 text-amber-200">
                            {product.category_name}
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-white text-slate-950 transition-transform group-hover:scale-105">
                      <Plus className="h-4 w-4" />
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="rounded-[30px] border border-white/10 bg-[#081120]/90 shadow-[0_30px_90px_rgba(0,0,0,0.32)]">
            <CardContent className="p-5 md:p-6">
              <div className="mb-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-slate-400">Saved Combos</p>
                  <h3 className="mt-2 text-2xl font-semibold text-white">Combo Catalog</h3>
                </div>
                <div className="rounded-full bg-white/10 px-4 py-2 text-sm text-slate-200">
                  {combos.length} combos available
                </div>
              </div>

              <div className="space-y-4">
                {combos.length === 0 && (
                  <div className="rounded-[26px] border border-dashed border-white/15 bg-white/5 px-6 py-10 text-center text-sm text-slate-300">
                    No combos created yet
                  </div>
                )}

                {combos.map((combo) => (
                  <div
                    key={combo.id}
                    className="rounded-[30px] border border-white/10 bg-[linear-gradient(135deg,rgba(20,33,61,0.9)_0%,rgba(15,23,42,0.95)_100%)] p-4"
                  >
                    <div className="flex flex-col gap-5 xl:flex-row xl:items-center xl:justify-between">
                      <div className="min-w-0 flex-1">
                        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
                          <div className="min-w-0">
                            <div className="flex items-center gap-2">
                              <div className="rounded-full bg-sky-500/15 p-2 text-sky-300">
                                <Boxes className="h-4 w-4" />
                              </div>
                              <h4 className="truncate text-xl font-semibold text-white">{combo.name}</h4>
                            </div>
                            <p className="mt-2 text-sm text-slate-300">
                              {combo.items?.length || 0} products inside this combo
                            </p>
                          </div>

                          <div className="rounded-2xl bg-white px-4 py-2 text-sm font-semibold text-slate-950">
                            Rs. {formatCurrency(combo.price)}
                          </div>
                        </div>

                        <div className="mt-4 overflow-x-auto pb-2">
                          <div className="flex min-w-max gap-3">
                            {(combo.items || []).map((item) => (
                              <div
                                key={`${combo.id}-${item.product_id}`}
                                className="w-[250px] rounded-2xl border border-white/10 bg-white/8 px-4 py-3"
                              >
                                <p className="truncate text-sm font-semibold text-white">{item.product_name}</p>
                                <p className="mt-1 text-xs text-slate-300">
                                  Qty: {item.quantity}
                                </p>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-2 xl:flex-col">
                        <Button
                          size="icon"
                          onClick={() => handleEdit(combo)}
                          className="h-11 w-11 rounded-2xl bg-white text-slate-950 hover:bg-slate-100"
                        >
                          <Pencil className="h-4 w-4" />
                        </Button>
                        <Button
                          size="icon"
                          variant="destructive"
                          onClick={() => deleteCombo(combo.id)}
                          className="h-11 w-11 rounded-2xl"
                        >
                          <Trash2 className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </section>
      </div>
    </div>
  )
}

export default Combo
