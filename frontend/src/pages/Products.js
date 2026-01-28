"use client"

import { RefreshCw } from "lucide-react"

import { useState, useEffect, useMemo } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger, DialogClose } from "@/components/ui/dialog"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { toast } from "sonner"
import { Plus, Edit, Trash2, Search, QrCode, Camera, X, ImageIcon } from "lucide-react"
import imageCompression from "browser-image-compression"
import { useNavigate } from "react-router-dom"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`
export const compressImage = async (file) => {
  const options = {
    maxSizeMB: 0.25,              // ⬅️ even safer (250 KB)
    maxWidthOrHeight: 1024,       // ⬅️ force smaller
    useWebWorker: true,
    fileType: "image/jpeg",
    initialQuality: 0.7,          // ⬅️ important
    exifOrientation: 1,           // ⬅️ FORCE FIX ROTATION
  }

  try {
    const compressedFile = await imageCompression(file, options)
    return compressedFile
  } catch (err) {
    console.error("Image compression failed", err)
    return file
  }
}



const Products = () => {
  const navigate = useNavigate()

  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => setIsMobile(window.innerWidth < 768)
    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  const user = JSON.parse(localStorage.getItem("user") || "{}")
  const role = user?.role

  const [products, setProducts] = useState([])
  const [categories, setCategories] = useState([])
  const [open, setOpen] = useState(false)
  const [editingProduct, setEditingProduct] = useState(null)
  const [savingProduct, setSavingProduct] = useState(false)

  const [search, setSearch] = useState("")
  const [categoryFilter, setCategoryFilter] = useState("all")
  const [priceSort, setPriceSort] = useState("none")
  const [uploading, setUploading] = useState(false)
  const [imagePreviews, setImagePreviews] = useState([])
  const [variantOpen, setVariantOpen] = useState(false)
  const [selectedProduct, setSelectedProduct] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [page, setPage] = useState(1)
  const [limit] = useState(10)
  const [totalPages, setTotalPages] = useState(1)

  const [variants, setVariants] = useState([
    { v_sku: "", variant_name: "", color: "", size: "", stock: 0, image_url: "" },
  ])
  const [formData, setFormData] = useState({
    name: "",
    description: "",
    category_id: "",
    cost_price: "",
    min_selling_price: "",
    selling_price: "",
    min_stock: "",
    sku: "",
    images: [],
  })
  const goToInventory = (sku, mode) => {
  navigate(`/inventory?sku=${encodeURIComponent(sku)}&mode=${mode}`)
}


  const servicesCategory = categories.find((c) => c.name.toLowerCase() === "services")
  const isServiceSelected = !!servicesCategory && formData.category_id === servicesCategory.id

  useEffect(() => {
    const token = localStorage.getItem("token")
    if (token) {
      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`
    }
  }, [])
  useEffect(() => {
    fetchProducts()
  }, [page])

  useEffect(() => {
    fetchCategories()
  }, [])
  useEffect(() => {
    setPage(1)
  }, [search, categoryFilter, priceSort])


  const fetchProducts = async () => {
    try {
      const res = await axios.get(`${API}/products`, {
        params: {
          page,
          limit,
        },
      })

      setProducts(res.data.data)
      setTotalPages(res.data.total_pages)
    } catch {
      toast.error("Failed to load products")
    }
  }
  const handleRefresh = async () => {
    try {
      setRefreshing(true)
      await fetchProducts()
      toast.success("Products refreshed")
    } catch {
      toast.error("Failed to refresh products")
    } finally {
      setRefreshing(false)
    }
  }

  const fetchCategories = async () => {
    try {
      const res = await axios.get(`${API}/categories`)
      setCategories(res.data)
    } catch {
      toast.error("Failed to load categories")
    }
  }
  const handleImageUpload = async (file) => {
    if (!file) return

    if (formData.images.length >= 5) {
      toast.error("Maximum 5 images allowed")
      return
    }

    if (!file.type.startsWith("image/")) {
      toast.error("Only image files allowed")
      return
    }

    if (file.size > 15 * 1024 * 1024) {
      toast.error("Image too large (max 15MB)")
      return
    }

    const token = localStorage.getItem("token")
    const compressed = await compressImage(file)

    const formDataUpload = new FormData()
    formDataUpload.append("file", compressed)

    try {
      setUploading(true)

      const res = await axios.post(
        `${API}/upload/product-image`,
        formDataUpload,
        {
          headers: {
            Authorization: `Bearer ${token}`, // ✅ NO Content-Type
          },
        }
      )

      setFormData((prev) => ({
        ...prev,
        images: [...prev.images, res.data.url].slice(0, 5),
      }))
      setImagePreviews((prev) => [...prev, res.data.url].slice(0, 5))

      toast.success("Image uploaded")
    } catch (err) {
      console.error(err)
      toast.error("Image upload failed")
    } finally {
      setUploading(false)
    }
  }
  const handleVariantImageUpload = async (file, index) => {
    if (!file) return

    if (!file.type.startsWith("image/")) {
      toast.error("Only image files allowed")
      return
    }

    if (file.size > 15 * 1024 * 1024) {
      toast.error("Image too large (max 15MB)")
      return
    }

    const token = localStorage.getItem("token")
    const compressed = await compressImage(file)

    const formDataUpload = new FormData()
    formDataUpload.append("file", compressed)

    try {
      const res = await axios.post(
        `${API}/upload/variant-image`,
        formDataUpload,
        {
          headers: {
            Authorization: `Bearer ${token}`, // ✅ NO Content-Type
          },
        }
      )

      const copy = [...variants]
      copy[index].image_url = res.data.url
      setVariants(copy)

      toast.success("Variant image uploaded")
    } catch (err) {
      console.error(err)
      toast.error("Variant image upload failed")
    }
  }

  const filteredProducts = useMemo(() => {
    let data = [...products]
    if (search.trim()) {
      const q = search.toLowerCase()
      data = data.filter(
        (p) =>
          p.name?.toLowerCase().includes(q) ||
          p.sku?.toLowerCase().includes(q) ||
          p.product_code?.toLowerCase().includes(q) ||
          p.category_name?.toLowerCase().includes(q),
      )
    }
    if (categoryFilter !== "all") data = data.filter((p) => p.category_id === categoryFilter)
    if (priceSort === "low-high") data.sort((a, b) => a.selling_price - b.selling_price)
    else if (priceSort === "high-low") data.sort((a, b) => b.selling_price - a.selling_price)
    else data.sort((a, b) => a.name.localeCompare(b.name))
    return data
  }, [products, search, categoryFilter, priceSort])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setSavingProduct(true) // ⬅️ START LOADING

    const hasValidVariants = variants.some((v) => v.v_sku && v.v_sku.trim() !== "")
    const finalVariants =
      isServiceSelected || !hasValidVariants
        ? []
        : variants
          .filter((v) => v.v_sku && v.v_sku.trim() !== "")
          .map((v) => ({
            v_sku: v.v_sku,
            variant_name: v.variant_name,
            color: v.color,
            size: v.size,
            image_url: v.image_url,
          }))

    const payload = {
      ...formData,
      description: formData.description || null,
      cost_price: Number(formData.cost_price || 0),
      selling_price: Number(formData.selling_price || 0),
      min_selling_price: Number(formData.min_selling_price || 0),
      min_stock: isServiceSelected ? 0 : Number(formData.min_stock || 0),
      is_service: isServiceSelected ? 1 : 0,
      variants: finalVariants,
      qr_code_url: null,
    }

    try {
      if (editingProduct) {
        await axios.put(`${API}/products/${editingProduct.id}`, payload)
      } else {
        await axios.post(`${API}/products`, payload)
      }

      toast.success(editingProduct ? "Product updated" : "Product created")
      fetchProducts()
      resetForm()
      setOpen(false)
    } catch (err) {
      toast.error(err.response?.data?.detail || "Failed to save product")
    } finally {
      setSavingProduct(false) // ⬅️ STOP LOADING
    }
  }


  const handleEdit = (product) => {
    setEditingProduct(product)
    const isService = product.is_service === 1
    setFormData({
      name: product.name || "",
      description: product.description || "",
      category_id: product.category_id || "",
      selling_price: product.selling_price?.toString() || "",
      min_selling_price: product.min_selling_price?.toString() || "",
      min_stock: isService ? "0" : product.min_stock?.toString() || "",
      sku: product.sku || "",
      images: Array.isArray(product.images) ? product.images : [],
      cost_price: role === "admin" ? product.cost_price?.toString() || "" : "",
    })
    setVariants(
      product.variants?.length
        ? product.variants.map((v) => ({ ...v, stock: Number(v.stock) }))
        : [{ v_sku: "", variant_name: "", color: "", size: "", stock: 0, image_url: "" }],
    )
    setImagePreviews(Array.isArray(product.images) ? product.images : [])
    setOpen(true)
  }

  const handleDelete = async (id) => {
    if (!window.confirm("Delete this product?")) return

    try {
      await axios.delete(`${API}/products/${id}`)
      toast.success("Product deleted")

      // ⬅️ If last item on page deleted, go back one page
      if (filteredProducts.length === 1 && page > 1) {
        setPage((p) => p - 1)
      } else {
        fetchProducts()
      }
    } catch {
      toast.error("Failed to delete product")
    }
  }


  const resetForm = () => {
    setEditingProduct(null)
    setImagePreviews([])
    setFormData({
      name: "",
      description: "",
      category_id: "",
      cost_price: "",
      min_selling_price: "",
      selling_price: "",
      stock: "0",
      min_stock: "",
      sku: "",
      images: [],
    })
    setVariants([{ v_sku: "", variant_name: "", color: "", size: "", stock: 0, image_url: "" }])
  }

  const removeImage = (index) => {
    const updated = formData.images.filter((_, i) => i !== index)
    setFormData((prev) => ({ ...prev, images: updated }))
    setImagePreviews(updated)
  }

  return (
    <div className="p-4 sm:p-6 space-y-6 bg-background min-h-screen">
      {/* HEADER */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl sm:text-4xl font-bold tracking-tight">Products</h1>
          <p className="text-muted-foreground text-sm sm:text-base">Manage your inventory</p>
        </div>

        {role === "admin" && (
          <div className="flex items-center gap-2">
            {/* 🔄 Refresh Button (OUTSIDE Dialog) */}
            <Button
              variant="outline"
              size={isMobile ? "sm" : "default"}
              onClick={handleRefresh}
              disabled={refreshing}
              className="rounded-full flex items-center gap-2"
              title="Refresh products"
            >
              <RefreshCw className={`w-4 h-4 ${refreshing ? "animate-spin" : ""}`} />
              <span className="hidden sm:inline">
                {refreshing ? "Refreshing..." : "Refresh"}
              </span>
            </Button>

            {/* ➕ Add Product Dialog */}
            <Dialog
              open={open}
              onOpenChange={setOpen}

            >
              <DialogTrigger asChild>
                <Button size={isMobile ? "sm" : "default"} className="rounded-full px-6">
                  <Plus className="w-4 h-4 mr-2" />
                  {isMobile ? "Add" : "Add Product"}
                </Button>
              </DialogTrigger>

              {/* ⬇️ KEEP EVERYTHING BELOW EXACTLY SAME */}
              <DialogContent
                className={`${isMobile
                  ? "w-full h-[100dvh] max-w-none m-0 rounded-none flex flex-col p-0 overflow-hidden"
                  : "max-w-2xl max-h-[90vh] overflow-y-auto"
                  }`}
              >
                <DialogHeader
                  className={isMobile ? "p-4 border-b flex flex-row items-center justify-between space-y-0" : ""}
                >
                  <DialogTitle className="text-xl font-bold">
                    {editingProduct ? "Edit Product" : "Add Product"}
                  </DialogTitle>
                  {isMobile && (
                    <DialogClose className="rounded-full p-2 hover:bg-muted transition-colors">
                      <X className="h-5 w-5" />
                    </DialogClose>
                  )}
                </DialogHeader>

                <form onSubmit={handleSubmit} className="flex flex-col h-full overflow-hidden">
                  <div className={`flex-1 overflow-y-auto p-4 space-y-6 ${isMobile ? "pb-32" : ""}`}>
                    {/* BASIC INFO */}
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                          Product Name
                        </Label>
                        <Input
                          value={formData.name}
                          onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                          required
                          className="bg-muted/30 border-muted focus-visible:ring-primary h-12"
                          placeholder="Enter product name"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">SKU</Label>
                        <Input
                          value={formData.sku}
                          onChange={(e) => setFormData({ ...formData, sku: e.target.value })}
                          required
                          className="bg-muted/30 border-muted focus-visible:ring-primary h-12"
                          placeholder="Enter SKU"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                        Description
                      </Label>
                      <Input
                        value={formData.description}
                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                        className="bg-muted/30 border-muted h-12"
                        placeholder="Optional description"
                      />
                    </div>

                    <div className="space-y-2">
                      <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">Category</Label>
                      <Select
                        value={formData.category_id}
                        onValueChange={(v) => {
                          const sel = categories.find((c) => c.id === v)
                          const isSrv = sel?.name?.toLowerCase() === "services"

                          setFormData({
                            ...formData,
                            category_id: v,
                            stock: isSrv ? "0" : formData.stock,
                            min_stock: isSrv ? "0" : formData.min_stock,
                          })
                        }}
                      >
                        <SelectTrigger className="bg-black text-white border border-border h-12">
                          <SelectValue placeholder="Select category" />
                        </SelectTrigger>

                        <SelectContent className="bg-black text-white border border-border">
                          {categories.map((c) => (
                            <SelectItem
                              key={c.id}
                              value={c.id}
                              className="focus:bg-white/10"
                            >
                              {c.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>

                    </div>

                    {/* IMAGES */}
                    <div className="space-y-3">
                      <div className="flex items-center justify-between">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                          Product Images
                        </Label>
                        <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-muted">
                          {formData.images.length}/5
                        </span>
                      </div>
                      {/* READ-ONLY CURRENT STOCK (EDIT MODE ONLY) */}
                      {editingProduct && !isServiceSelected && (
                        <div className="flex items-center gap-2 px-1">
                          <span className="text-xs uppercase tracking-wider font-bold text-muted-foreground">
                            Current Stock
                          </span>

                          <span
                            className={`text-xs font-black px-2 py-1 rounded-full ${editingProduct.stock <= editingProduct.min_stock
                                ? "bg-destructive/10 text-destructive"
                                : "bg-green-500/10 text-green-500"
                              }`}
                          >
                            {editingProduct.stock}
                          </span>

                          {editingProduct.variants?.length > 0 && (
                            <span className="text-[10px] font-bold text-muted-foreground">
                              ({editingProduct.variants.length} variants)
                            </span>
                          )}
                        </div>
                      )}

                      <div className="grid grid-cols-2 gap-3">
                        <Label className="flex flex-col items-center justify-center h-28 rounded-2xl border-2 border-dashed border-muted hover:border-primary/50 transition-all bg-muted/10 cursor-pointer active:scale-95">
                          <Camera className="w-6 h-6 mb-2 text-primary" />
                          <span className="text-xs font-bold">Snap Photo</span>
                          <Input
                            type="file"
                            accept="image/*"
                            capture="environment"
                            className="hidden"
                            onChange={(e) => handleImageUpload(e.target.files[0])}
                            disabled={uploading || formData.images.length >= 5}
                          />
                        </Label>
                        <Label className="flex flex-col items-center justify-center h-28 rounded-2xl border-2 border-dashed border-muted hover:border-primary/50 transition-all bg-muted/10 cursor-pointer active:scale-95">
                          <ImageIcon className="w-6 h-6 mb-2 text-primary" />
                          <span className="text-xs font-bold">From Gallery</span>
                          <Input
                            type="file"
                            accept="image/*"
                            className="hidden"
                            onChange={(e) => handleImageUpload(e.target.files[0])}
                            disabled={uploading || formData.images.length >= 5}
                          />
                        </Label>
                      </div>

                      <div className="flex gap-3 overflow-x-auto py-2 scrollbar-hide">
                        {imagePreviews.map((img, idx) => (
                          <div key={idx} className="relative flex-shrink-0">
                            <img
                              src={img || "/placeholder.svg"}
                              className={`w-24 h-24 object-cover rounded-2xl border-2 ${idx === 0 ? "border-primary ring-4 ring-primary/10" : "border-muted"}`}
                            />
                            <button
                              type="button"
                              onClick={() => removeImage(idx)}
                              className="absolute -top-2 -right-2 bg-destructive text-white rounded-full p-1.5 shadow-xl active:scale-90 transition-transform"
                            >
                              <X className="w-3 h-3" />
                            </button>
                            {idx === 0 && (
                              <span className="absolute bottom-2 left-2 bg-primary text-white text-[8px] px-1.5 py-0.5 font-black rounded-md tracking-tighter">
                                MAIN
                              </span>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* PRICING & STOCK */}
                    <div className="grid grid-cols-2 gap-4 pt-2">
                      {role === "admin" && (
                        <div className="space-y-2">
                          <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                            Cost Price
                          </Label>
                          <Input
                            type="number"
                            value={formData.cost_price}
                            onChange={(e) => setFormData({ ...formData, cost_price: e.target.value })}
                            className="bg-muted/30 border-muted h-12"
                          />
                        </div>
                      )}
                      <div className="space-y-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                          Min Selling Price
                        </Label>
                        <Input
                          type="number"
                          value={formData.min_selling_price}
                          onChange={(e) => setFormData({ ...formData, min_selling_price: e.target.value })}
                          className="bg-muted/30 border-muted h-12"
                        />
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                          Selling Price
                        </Label>
                        <Input
                          type="number"
                          value={formData.selling_price}
                          onChange={(e) => setFormData({ ...formData, selling_price: e.target.value })}
                          className="bg-muted/30 border-muted h-12 font-bold text-primary"
                        />
                      </div>
                      {!isServiceSelected && (
                        <>

                          <div className="space-y-2">
                            <Label className="text-xs uppercase tracking-wider text-muted-foreground font-bold">
                              Low Stock Alert
                            </Label>
                            <Input
                              type="number"
                              value={formData.min_stock}
                              onChange={(e) => setFormData({ ...formData, min_stock: e.target.value })}
                              className="bg-muted/30 border-muted h-12"
                            />
                          </div>
                        </>
                      )}
                    </div>

                    {/* VARIANTS SECTION */}
                    {!isServiceSelected && (
                      <div className="space-y-4 pt-6 border-t">
                        <div className="flex items-center justify-between">
                          <Label className="text-sm font-black uppercase tracking-widest text-primary">Variants</Label>
                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            className="rounded-full h-8"
                            onClick={() =>
                              setVariants([
                                ...variants,
                                { v_sku: "", variant_name: "", color: "", size: "", stock: 0, image_url: "" },
                              ])
                            }
                          >
                            <Plus className="w-3 h-3 mr-1" /> Add Variant
                          </Button>
                        </div>

                        <div className="space-y-4">
                          {variants.map((v, idx) => (
                            <Card key={idx} className="relative overflow-hidden border-muted/50 shadow-sm">
                              <button
                                type="button"
                                onClick={() => setVariants(variants.filter((_, i) => i !== idx))}
                                disabled={variants.length === 1}
                                className="absolute top-3 right-3 p-1.5 text-muted-foreground hover:text-destructive bg-muted/50 rounded-full"
                              >
                                <Trash2 className="w-4 h-4" />
                              </button>

                              <CardContent className="p-4 space-y-4">
                                <div className="grid grid-cols-2 gap-3">
                                  <div className="space-y-1.5">
                                    <Label className="text-[10px] uppercase font-bold text-muted-foreground">V-SKU</Label>
                                    <Input
                                      placeholder="e.g. SKU-RED-M"
                                      value={v.v_sku}
                                      onChange={(e) => {
                                        const copy = [...variants]
                                        copy[idx].v_sku = e.target.value
                                        setVariants(copy)
                                      }}
                                      className="h-10 text-xs"
                                    />
                                  </div>
                                  <div className="space-y-1.5">
                                    <Label className="text-[10px] uppercase font-bold text-muted-foreground">Name</Label>
                                    <Input
                                      placeholder="Variant name"
                                      value={v.variant_name}
                                      onChange={(e) => {
                                        const copy = [...variants]
                                        copy[idx].variant_name = e.target.value
                                        setVariants(copy)
                                      }}
                                      className="h-10 text-xs"
                                    />
                                  </div>
                                </div>

                                <div className="grid grid-cols-3 gap-3">
                                  <div className="space-y-1.5">
                                    <Label className="text-[10px] uppercase font-bold text-muted-foreground">Color</Label>
                                    <Input
                                      placeholder="Color"
                                      value={v.color}
                                      onChange={(e) => {
                                        const copy = [...variants]
                                        copy[idx].color = e.target.value
                                        setVariants(copy)
                                      }}
                                      className="h-10 text-xs"
                                    />
                                  </div>
                                  <div className="space-y-1.5">
                                    <Label className="text-[10px] uppercase font-bold text-muted-foreground">Size</Label>
                                    <Input
                                      placeholder="Size"
                                      value={v.size}
                                      onChange={(e) => {
                                        const copy = [...variants]
                                        copy[idx].size = e.target.value
                                        setVariants(copy)
                                      }}
                                      className="h-10 text-xs"
                                    />
                                  </div>
                                  {/* <div className="space-y-1.5">
                                    <Label className="text-[10px] uppercase font-bold text-muted-foreground">Stock</Label>
                                    <Input
                                      type="number"
                                      value={v.stock}
                                      onChange={(e) => {
                                        const copy = [...variants]
                                        copy[idx].stock = e.target.value
                                        setVariants(copy)
                                      }}
                                      className="h-10 text-xs"
                                    />
                                  </div> */}
                                </div>

                                <div className="flex items-center gap-4 pt-2">
                                  <Label className="flex-1 flex items-center justify-center gap-2 h-10 border-2 border-dashed rounded-xl cursor-pointer hover:bg-muted/20 text-xs font-bold">
                                    <Camera className="w-4 h-4" />
                                    {v.image_url ? "Change Image" : "Add Image"}
                                    <Input
                                      type="file"
                                      accept="image/*"
                                      capture="environment"
                                      className="hidden"
                                      onChange={(e) => handleVariantImageUpload(e.target.files[0], idx)}
                                    />
                                  </Label>
                                  {v.image_url && (
                                    <img
                                      src={v.image_url || "/placeholder.svg"}
                                      className="w-10 h-10 rounded-lg object-cover ring-2 ring-primary/20"
                                    />
                                  )}
                                </div>
                              </CardContent>
                            </Card>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* STICKY ACTION BUTTONS FOR MOBILE */}
                  <div
                    className={`${isMobile ? "fixed bottom-0 left-0 right-0 p-4 bg-background border-t grid grid-cols-2 gap-3 z-50" : "p-4 flex justify-end gap-3 border-t"}`}
                  >
                    <Button
                      type="button"
                      variant="outline"
                      className="h-14 sm:h-10 rounded-2xl font-bold bg-transparent"
                      onClick={() => {
                        setOpen(false)
                        resetForm()
                      }}
                    >
                      Cancel
                    </Button>
                    <Button
                      type="submit"
                      disabled={savingProduct || uploading}
                      className="h-14 sm:h-10 rounded-2xl font-bold shadow-lg flex items-center justify-center gap-2"
                    >
                      {savingProduct && (
                        <span className="h-4 w-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      )}
                      {savingProduct
                        ? "Saving..."
                        : editingProduct
                          ? "Update Product"
                          : "Create Product"}
                    </Button>


                  </div>
                </form>
              </DialogContent>
            </Dialog>
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row gap-4">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search products..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10 h-11"
          />
        </div>
        <div className="flex gap-2">
          <Select value={categoryFilter} onValueChange={setCategoryFilter}>
            <SelectTrigger className="flex-1 sm:w-48 h-11 bg-black text-white border border-border">
              <SelectValue placeholder="Category" />
            </SelectTrigger>

            <SelectContent className="bg-black text-white border border-border">
              <SelectItem value="all" className="focus:bg-white/10">
                All categories
              </SelectItem>

              {categories.map((c) => (
                <SelectItem
                  key={c.id}
                  value={c.id}
                  className="focus:bg-white/10"
                >
                  {c.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Select value={priceSort} onValueChange={setPriceSort}>
            <SelectTrigger className="flex-1 sm:w-48 h-11 bg-black text-white border border-border">
              <SelectValue placeholder="Sort" />
            </SelectTrigger>

            <SelectContent className="bg-black text-white border border-border">
              <SelectItem value="none" className="focus:bg-white/10">
                Default Sort
              </SelectItem>
              <SelectItem value="low-high" className="focus:bg-white/10">
                Price: Low to High
              </SelectItem>
              <SelectItem value="high-low" className="focus:bg-white/10">
                Price: High to Low
              </SelectItem>
            </SelectContent>
          </Select>

        </div>
      </div>

      {/* MOBILE LIST VIEW */}
      {isMobile ? (
        <div className="space-y-4">
          {filteredProducts.map((p) => (
            <Card key={p.id} className="overflow-hidden border-none shadow-sm bg-muted/20">
              <div className="flex gap-4 p-4">
                <div className="relative">
                  <img
                    src={p.images?.[0] || "/placeholder.svg"}
                    className="w-20 h-20 object-cover rounded-2xl shadow-sm bg-background"
                    alt={p.name}
                  />
                  <div className="absolute -top-1 -right-1">
                    {p.qr_code_url && (!p.variants || p.variants.length === 0) && (
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button size="icon" variant="secondary" className="w-7 h-7 rounded-full shadow-md scale-90">
                            <QrCode className="w-3.5 h-3.5" />
                          </Button>
                        </DialogTrigger>

                        <DialogContent className="max-w-[300px] rounded-3xl">
                          <DialogHeader>
                            <DialogTitle className="text-center font-bold">{p.name} QR</DialogTitle>
                          </DialogHeader>
                          <div className="flex flex-col items-center gap-4 p-4">
                            <div className="p-4 bg-white rounded-2xl shadow-inner">
                              <img src={p.qr_code_url} className="w-40 h-40" />
                            </div>
                            <span className="font-mono text-xs font-bold text-muted-foreground tracking-widest">
                              {p.sku}
                            </span>
                          </div>
                        </DialogContent>
                      </Dialog>
                    )}

                  </div>
                </div>

                <div className="flex-1 flex flex-col justify-between py-0.5">
                  <div>
                    <h3 className="font-bold text-base leading-tight line-clamp-1">{p.name}</h3>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase tracking-wider mt-1">
                      {p.category_name} • {p.sku}
                    </p>
                  </div>

                  <div className="flex items-center justify-between mt-auto">
                    <span className="text-lg font-black text-primary">₹{p.selling_price}</span>
                    <div className="flex items-center gap-1.5">
                      <span
                        className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${p.stock <= p.min_stock
                          ? "bg-destructive/10 text-destructive"
                          : "bg-green-500/10 text-green-500"
                          }`}
                      >
                        {p.stock} in stock
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="flex items-center border-t border-muted p-2 gap-2 bg-muted/10">
                <Button
                  variant="ghost"
                  size="sm"
                  className="flex-1 h-10 rounded-xl font-bold text-xs"
                  onClick={() => handleEdit(p)}
                >
                  <Edit className="w-3.5 h-3.5 mr-2" /> Edit
                </Button>
                          <Button
  variant="outline"
  size="sm"
  className="h-10 rounded-xl text-xs font-bold"
  onClick={() => goToInventory(p.sku, "in")}
>
  IN
</Button>

<Button
  variant="outline"
  size="sm"
  className="h-10 rounded-xl text-xs font-bold"
  onClick={() => goToInventory(p.sku, "out")}
>
  OUT
</Button>
                {p.variants?.length > 0 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="flex-1 h-10 rounded-xl font-bold text-xs"
                    onClick={() => {
                      setSelectedProduct(p)
                      setVariantOpen(true)
                    }}
                  >
                    {p.variants.length} Variants
                  </Button>
                )}

                {role === "admin" && (
                  <Button
                    variant="ghost"
                    size="sm"
                    className="w-10 h-10 rounded-xl text-destructive hover:bg-destructive/10"
                    onClick={() => handleDelete(p.id)}
                  >
                    <Trash2 className="w-4 h-4" />
                  </Button>
                )}
              </div>
            </Card>

          ))}
          {isMobile && (
            <div className="flex items-center justify-center gap-4 pt-6">
              <Button
                variant="outline"
                disabled={page === 1 || refreshing}
                onClick={() => setPage((p) => p - 1)}
              >
                Previous
              </Button>

              <span className="text-sm font-bold">
                Page {page} of {totalPages}
              </span>

              <Button
                variant="outline"
                disabled={page === totalPages || refreshing}
                onClick={() => setPage((p) => p + 1)}
              >
                Next
              </Button>
            </div>
          )}

        </div>
      ) : (
        // DESKTOP TABLE VIEW
        <Card className="border-muted shadow-lg overflow-hidden">
          <CardHeader className="bg-muted/30 border-b">
            <CardTitle className="flex items-center gap-2">
              <span className="w-2 h-6 bg-primary rounded-full" />
              Product Inventory (Page {page} of {totalPages})
            </CardTitle>
          </CardHeader>
          <CardContent className="p-0 overflow-x-auto">
            <table className="w-full border-collapse">
              <thead>
                <tr className="bg-muted/50 text-xs font-bold text-muted-foreground uppercase tracking-wider border-b">
                  <th className="p-4 text-left">#</th>
                  <th className="p-4 text-left">Name / Info</th>
                  <th className="p-4 text-left">Category</th>
                  <th className="p-4 text-left">Image</th>
                  <th className="p-4 text-left">SKU</th>
                  <th className="p-4 text-left text-center">QR</th>
                  {role === "admin" && <th className="p-4 text-left">Cost</th>}
                  <th className="p-4 text-left">Sell Price</th>
                  <th className="p-4 text-left">Stock</th>
                  <th className="p-4 text-left">Variants</th>
                  {role === "admin" && <th className="p-4 text-center">Actions</th>}
                </tr>
              </thead>
              <tbody className="divide-y divide-muted/30">
                {filteredProducts.map((p, i) => (
                  <tr key={p.id} className="hover:bg-muted/20 transition-colors group">
                    <td className="p-4 text-sm text-muted-foreground">
                      {(page - 1) * limit + i + 1}
                    </td>                    <td className="p-4">
                      <div className="flex flex-col">
                        <span className="font-bold text-sm">{p.name}</span>
                        <span className="text-[10px] font-mono text-muted-foreground mt-1">{p.product_code}</span>
                      </div>
                    </td>
                    <td className="p-4">
                      <span className="text-xs font-semibold px-2 py-1 bg-muted rounded-md">{p.category_name}</span>
                    </td>
                    <td className="p-4">
                      {p.images?.[0] ? (
                        <Dialog>
                          <DialogTrigger asChild>
                            <img
                              src={p.images[0] || "/placeholder.svg"}
                              className="w-10 h-10 object-cover rounded-lg border hover:scale-110 transition-transform cursor-zoom-in shadow-sm"
                            />
                          </DialogTrigger>
                          <DialogContent className="max-w-2xl">
                            <div className="flex gap-4 overflow-x-auto p-2">
                              {p.images.map((img, idx) => (
                                <img
                                  key={idx}
                                  src={img || "/placeholder.svg"}
                                  className="h-64 w-auto rounded-xl border-2 object-cover"
                                />
                              ))}
                            </div>
                          </DialogContent>
                        </Dialog>
                      ) : (
                        <span className="text-xs text-muted-foreground italic">None</span>
                      )}
                    </td>
                    <td className="p-4 font-mono text-xs">{p.sku}</td>
                    <td className="p-4 text-center">
                      {p.qr_code_url && (!p.variants || p.variants.length === 0) ? (
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button
                              size="icon"
                              variant="outline"
                              className="h-8 w-8 hover:bg-primary hover:text-primary-foreground transition-all bg-transparent"
                            >
                              <QrCode className="w-4 h-4" />
                            </Button>
                          </DialogTrigger>

                          <DialogContent className="max-w-sm">
                            <div className="flex flex-col items-center gap-4 py-4">
                              <img
                                src={p.qr_code_url}
                                alt="QR"
                                className="w-48 h-48 border-2 p-2 bg-white rounded-lg"
                              />
                              <p className="font-mono text-xs font-bold">{p.sku}</p>
                              <Button
                                className="w-full"
                                onClick={() => window.open(p.qr_code_url, "_blank")}
                              >
                                Download QR
                              </Button>
                            </div>
                          </DialogContent>
                        </Dialog>
                      ) : (
                        <span className="text-xs text-muted-foreground">-</span>
                      )}
                    </td>


                    {role === "admin" && <td className="p-4 text-sm font-medium">₹{p.cost_price}</td>}
                    <td className="p-4 text-sm font-bold text-primary">₹{p.selling_price}</td>
                    <td className="p-4">
                      <div className="flex items-center gap-2">
                        <span
                          className={`text-sm font-bold ${p.stock <= p.min_stock ? "text-destructive" : "text-green-500"}`}
                        >
                          {p.stock}
                        </span>
                        {p.stock <= p.min_stock && (
                          <span className="h-2 w-2 rounded-full bg-destructive animate-pulse" />
                        )}
                      </div>
                    </td>
                    <td className="p-4">
                      {p.variants?.length > 0 ? (
                        <Button
                          size="sm"
                          variant="ghost"
                          className="text-xs font-bold hover:bg-primary/10 hover:text-primary"
                          onClick={() => {
                            setSelectedProduct(p)
                            setVariantOpen(true)
                          }}
                        >
                          {p.variants.length} Vars
                        </Button>
                      ) : (
                        <span className="text-xs text-muted-foreground">-</span>
                      )}
                    </td>
                    {role === "admin" && (
                      <td className="p-4">
                        <div className="flex items-center justify-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Button
                            size="icon"
                            variant="ghost"
                            title="Stock In"
                            onClick={() => goToInventory(p.sku, "in")}
                          >
                            ⬆️
                          </Button>

                          <Button
                            size="icon"
                            variant="ghost"
                            title="Stock Out"
                            onClick={() => goToInventory(p.sku, "out")}
                          >
                            ⬇️
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-8 w-8 text-muted-foreground hover:text-primary"
                            onClick={() => handleEdit(p)}
                          >
                            <Edit className="w-4 h-4" />
                          </Button>
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-8 w-8 text-muted-foreground hover:text-destructive"
                            onClick={() => handleDelete(p.id)}
                          >
                            <Trash2 className="w-4 h-4" />
                          </Button>
                        </div>
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          </CardContent>
        </Card>
      )}
      {!isMobile && (
        <div className="flex items-center justify-center gap-4 pt-6">
          <Button
            variant="outline"
            disabled={page === 1 || refreshing}
            onClick={() => setPage((p) => p - 1)}
          >
            Previous
          </Button>

          <span className="text-sm font-bold">
            Page {page} of {totalPages}
          </span>

          <Button
            variant="outline"
            disabled={page === totalPages || refreshing}
            onClick={() => setPage((p) => p + 1)}
          >
            Next
          </Button>
        </div>
      )}

      {/* VARIANT DRAWER/DIALOG */}
      <Dialog open={variantOpen} onOpenChange={setVariantOpen}>
        <DialogContent
          className={`${isMobile ? "w-full h-[100dvh] max-w-none m-0 rounded-none overflow-y-auto" : "max-w-3xl"}`}
        >
          <DialogHeader className={isMobile ? "sticky top-0 bg-background z-10 pb-4 border-b" : ""}>
            <DialogTitle className="font-black text-xl">Variants: {selectedProduct?.name}</DialogTitle>
          </DialogHeader>

          <div className={`space-y-4 ${isMobile ? "py-4 pb-12" : "py-4"}`}>
            {selectedProduct?.variants?.map((v, i) => (
              <Card key={i} className="p-4 border-none shadow-sm bg-muted/20">
                <div className="flex gap-4">
                  <div className="relative">
                    <img src={v.image_url || "/placeholder.svg"} className="w-16 h-16 object-cover rounded-xl" />
                    <div className="absolute -top-1 -right-1">
                      {v.qr_code_url && (
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button size="icon" variant="secondary" className="w-6 h-6 rounded-full shadow-md scale-90">
                              <QrCode className="w-3 h-3" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="max-w-[280px] rounded-3xl">
                            <DialogHeader>
                              <DialogTitle className="text-center text-sm font-bold">Variant QR: {v.v_sku}</DialogTitle>
                            </DialogHeader>
                            <div className="flex flex-col items-center gap-4 p-4">
                              <div className="p-3 bg-white rounded-2xl shadow-inner">
                                <img src={v.qr_code_url || "/placeholder.svg"} className="w-32 h-32" />
                              </div>
                              <span className="font-mono text-[10px] font-bold text-muted-foreground uppercase">
                                {v.v_sku}
                              </span>
                            </div>
                          </DialogContent>
                        </Dialog>
                      )}
                    </div>
                  </div>

                  <div className="flex-1 flex flex-col justify-center">
                    <h4 className="font-bold text-sm">
                      {v.variant_name || "Standard"} {v.color ? `(${v.color})` : ""} {v.size ? `• ${v.size}` : ""}
                    </h4>
                    <p className="text-[10px] font-bold text-muted-foreground uppercase mt-1">SKU: {v.v_sku}</p>
                    <div className="mt-2">
                      <span
                        className={`text-[10px] font-black px-2 py-0.5 rounded-full ${v.stock <= 0 ? "bg-destructive/10 text-destructive" : "bg-green-500/10 text-green-500"
                          }`}
                      >
                        {v.stock} in stock
                      </span>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}

export default Products
