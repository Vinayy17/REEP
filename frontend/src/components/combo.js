"use client"

import { useEffect, useMemo, useState } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue
} from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Plus, Trash2, Pencil, Package2, Search, Layers3, IndianRupee } from "lucide-react"
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
        price: Number(product.selling_price || 0),
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
        quantity: item.quantity,
        price: Number(item.price || 0)
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

  const suggestedTotal = useMemo(
    () => selectedItems.reduce((sum, item) => sum + (Number(item.price || 0) * Number(item.quantity || 0)), 0),
    [selectedItems]
  )

  return (
    <div className="h-full min-h-0 bg-background p-3 md:p-6">
      <div className="mx-auto grid max-w-7xl gap-4 xl:h-[calc(92vh-5rem)] xl:grid-cols-[380px_minmax(0,1fr)]">
        <Card className="border-border/60 shadow-sm xl:flex xl:min-h-0 xl:flex-col">
          <CardHeader className="space-y-4 border-b bg-muted/20">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Combo Builder</p>
                <CardTitle className="mt-2 text-2xl">
                  {editingComboId ? "Update Combo" : "Create Combo"}
                </CardTitle>
                <p className="mt-2 text-sm text-muted-foreground">
                  Build tap-friendly bundles with clear pricing and lightweight product selection.
                </p>
              </div>
              <div className="rounded-2xl bg-primary/10 p-3 text-primary">
                <Layers3 className="h-5 w-5" />
              </div>
            </div>

            <div className="grid grid-cols-3 gap-3">
              <div className="rounded-2xl bg-background p-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Lines</p>
                <p className="mt-1 text-lg font-bold">{selectedItems.length}</p>
              </div>
              <div className="rounded-2xl bg-background p-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Qty</p>
                <p className="mt-1 text-lg font-bold">{selectedSummary}</p>
              </div>
              <div className="rounded-2xl bg-background p-3">
                <p className="text-[11px] uppercase tracking-[0.18em] text-muted-foreground">Value</p>
                <p className="mt-1 text-lg font-bold">₹{formatCurrency(suggestedTotal)}</p>
              </div>
            </div>
          </CardHeader>

          <CardContent className="space-y-4 p-4 md:p-5 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
            <div className="grid gap-3">
              <Input
                value={comboName}
                onChange={(e) => setComboName(e.target.value)}
                placeholder="Combo name"
                className="h-12 rounded-2xl"
              />
              <div className="relative">
                <IndianRupee className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={comboPrice}
                  onChange={(e) => setComboPrice(e.target.value)}
                  placeholder="Combo price"
                  type="number"
                  className="h-12 rounded-2xl pl-10"
                />
              </div>
            </div>

            <div className="rounded-3xl border bg-muted/20 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold">Selected Products</p>
                  <p className="text-xs text-muted-foreground">Adjust quantity before saving</p>
                </div>
                <Badge variant="secondary" className="rounded-full px-3 py-1">
                  {selectedSummary} units
                </Badge>
              </div>

              <div className="space-y-3">
                {selectedItems.length === 0 && (
                  <div className="rounded-2xl border border-dashed bg-background/80 px-4 py-8 text-center text-sm text-muted-foreground">
                    No products selected yet
                  </div>
                )}

                {selectedItems.map((item) => (
                  <div key={item.product_id} className="rounded-2xl bg-background p-3 shadow-sm">
                    <div className="flex items-start gap-3">
                      <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                        <Package2 className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold">{item.name}</p>
                        <p className="text-xs text-muted-foreground">{item.sku || "No SKU"}</p>
                      </div>
                      <Button
                        size="icon"
                        variant="ghost"
                        onClick={() => removeItem(item.product_id)}
                        className="h-10 w-10 rounded-xl text-destructive hover:bg-destructive/10"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>

                    <div className="mt-3 flex items-center gap-3">
                      <Input
                        type="number"
                        min="1"
                        value={item.quantity}
                        onChange={(e) => updateQty(item.product_id, e.target.value)}
                        className="h-11 w-24 rounded-xl text-center"
                      />
                      <div className="rounded-xl bg-muted/40 px-3 py-2 text-sm font-semibold text-muted-foreground">
                        ₹{formatCurrency(item.price)} each
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
              <Button onClick={saveCombo} className="h-12 rounded-2xl">
                {editingComboId ? "Update Combo" : "Create Combo"}
              </Button>
              {editingComboId && (
                <Button variant="outline" onClick={resetForm} className="h-12 rounded-2xl">
                  Cancel
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="space-y-4 xl:min-h-0 xl:overflow-hidden">
          <Card className="border-border/60 shadow-sm xl:flex xl:min-h-0 xl:flex-col">
            <CardHeader className="space-y-4 border-b bg-muted/20">
              <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Products</p>
                  <CardTitle className="mt-2 text-2xl">Add Products</CardTitle>
                  <p className="mt-2 text-sm text-muted-foreground">
                    Search, filter, and tap any product card to add it to the combo.
                  </p>
                </div>
                <Badge variant="outline" className="w-fit rounded-full px-4 py-2 text-sm">
                  {products.length} products
                </Badge>
              </div>

              <div className="grid gap-3 md:grid-cols-[minmax(0,1fr)_220px]">
                <div className="relative">
                  <Search className="pointer-events-none absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
                  <Input
                    placeholder="Search products"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                    className="h-12 rounded-2xl pl-11"
                  />
                </div>

                <Select value={categoryFilter} onValueChange={setCategoryFilter}>
                  <SelectTrigger className="h-12 rounded-2xl">
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
            </CardHeader>

            <CardContent className="p-4 md:p-5 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
              <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-3">
                {products.map((product) => (
                  <button
                    key={product.id}
                    type="button"
                    onClick={() => addToCombo(product)}
                    className="rounded-3xl border bg-card p-4 text-left transition-all duration-200 hover:-translate-y-0.5 hover:border-primary/40 hover:shadow-md"
                  >
                    <div className="flex items-start gap-3">
                      <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-2xl bg-primary/10 text-primary">
                        <Package2 className="h-5 w-5" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <p className="truncate text-sm font-semibold">{product.name}</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Badge variant="secondary" className="rounded-full">
                            {product.sku || "No SKU"}
                          </Badge>
                          {product.category_name && (
                            <Badge variant="outline" className="rounded-full">
                              {product.category_name}
                            </Badge>
                          )}
                        </div>
                        <p className="mt-3 text-base font-bold text-primary">
                          ₹{formatCurrency(product.selling_price)}
                        </p>
                      </div>
                      <div className="flex h-10 w-10 flex-shrink-0 items-center justify-center rounded-full bg-primary text-primary-foreground">
                        <Plus className="h-4 w-4" />
                      </div>
                    </div>
                  </button>
                ))}
              </div>
            </CardContent>
          </Card>

          <Card className="border-border/60 shadow-sm xl:flex xl:min-h-0 xl:flex-col">
            <CardHeader className="border-b bg-muted/20">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <p className="text-xs uppercase tracking-[0.22em] text-muted-foreground">Saved Combos</p>
                  <CardTitle className="mt-2 text-2xl">Combo Catalog</CardTitle>
                </div>
                <Badge variant="outline" className="w-fit rounded-full px-4 py-2 text-sm">
                  {combos.length} combos
                </Badge>
              </div>
            </CardHeader>

            <CardContent className="space-y-4 p-4 md:p-5 xl:min-h-0 xl:flex-1 xl:overflow-y-auto">
              {combos.length === 0 && (
                <div className="rounded-3xl border border-dashed bg-muted/10 px-6 py-12 text-center text-sm text-muted-foreground">
                  No combos created yet
                </div>
              )}

              {combos.map((combo) => (
                <div key={combo.id} className="rounded-3xl border bg-card p-4 shadow-sm">
                  <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                        <div className="min-w-0">
                          <h4 className="truncate text-lg font-semibold">{combo.name}</h4>
                          <p className="mt-1 text-sm text-muted-foreground">
                            {combo.items?.length || 0} line items
                          </p>
                        </div>
                        <div className="rounded-2xl bg-primary/10 px-4 py-2 text-sm font-semibold text-primary">
                          ₹{formatCurrency(combo.price)}
                        </div>
                      </div>

                      <div className="mt-4 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                        {(combo.items || []).map((item) => (
                          <div key={`${combo.id}-${item.product_id}`} className="rounded-2xl bg-muted/20 px-4 py-3">
                            <p className="truncate text-sm font-semibold">{item.product_name}</p>
                            <p className="mt-1 text-xs text-muted-foreground">Qty: {item.quantity}</p>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-2 lg:grid-cols-1">
                      <Button
                        size="icon"
                        variant="outline"
                        onClick={() => handleEdit(combo)}
                        className="h-11 w-11 rounded-2xl"
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
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

export default Combo
