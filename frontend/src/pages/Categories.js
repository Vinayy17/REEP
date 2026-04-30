"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { useLocation, useNavigate } from "react-router-dom"
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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { toast } from "sonner"
import { Plus, FolderTree, Trash2, Search } from "lucide-react"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

export default function Categories() {
  const navigate = useNavigate()
  const location = useLocation()
  const [categories, setCategories] = useState([])
  const [open, setOpen] = useState(false)
  const [searchTerm, setSearchTerm] = useState("")
  const [formData, setFormData] = useState({
    name: "",
    description: "",
  })

  useEffect(() => {
    fetchCategories()
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get("action") === "add") {
      setOpen(true)
      navigate("/categories", { replace: true })
    }
  }, [location.search, navigate])

  const fetchCategories = async () => {
    try {
      const res = await axios.get(`${API}/categories`)
      setCategories(res.data)
    } catch {
      toast.error("Failed to load categories")
    }
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      await axios.post(`${API}/categories`, formData)
      toast.success("Category created successfully")
      fetchCategories()
      resetForm()
      setOpen(false)
    } catch (error) {
      toast.error(error.response?.data?.detail || "Failed to create category")
    }
  }

  const handleDelete = async (id) => {
    if (!window.confirm("Are you sure you want to delete this category?")) return
    try {
      await axios.delete(`${API}/categories/${id}`)
      toast.success("Category deleted successfully")
      fetchCategories()
    } catch {
      toast.error("Failed to delete category")
    }
  }

  const resetForm = () => {
    setFormData({ name: "", description: "" })
  }

  const filteredCategories = categories.filter((cat) =>
    `${cat.name} ${cat.description || ""}`.toLowerCase().includes(searchTerm.toLowerCase())
  )

  return (
    <div className="space-y-4 p-4 sm:p-6">
      <div className="flex justify-end">
        <Dialog
          open={open}
          onOpenChange={(next) => {
            setOpen(next)
            if (!next) resetForm()
          }}
        >
          <DialogTrigger asChild>
            <Button className="hidden rounded-2xl sm:inline-flex">
              <Plus className="mr-2 h-4 w-4" />
              Add Category
            </Button>
          </DialogTrigger>

          <DialogContent className="max-w-[95vw] rounded-3xl sm:max-w-md">
            <DialogHeader>
              <DialogTitle>Add New Category</DialogTitle>
            </DialogHeader>

            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label>Category Name</Label>
                <Input
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  required
                  className="h-12 rounded-2xl"
                />
              </div>

              <div className="space-y-2">
                <Label>Description</Label>
                <Input
                  value={formData.description}
                  onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                  className="h-12 rounded-2xl"
                />
              </div>

              <Button type="submit" className="h-12 w-full rounded-2xl">
                Create Category
              </Button>
            </form>
          </DialogContent>
        </Dialog>
      </div>

      <div className="relative">
        <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
        <Input
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search category..."
          className="h-10 rounded-2xl pl-10"
        />
      </div>

      <Card className="border-border/60 shadow-sm">
        <CardContent className="p-3 sm:p-4">
          {filteredCategories.length === 0 ? (
            <div className="py-12 text-center text-muted-foreground">
              <FolderTree className="mx-auto mb-3 h-10 w-10" />
              No categories found
            </div>
          ) : (
            <div className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
              {filteredCategories.map((cat, index) => (
                <Card key={cat.id} className="rounded-2xl border shadow-sm">
                  <CardContent className="space-y-3 p-3">
                    <div className="flex items-start gap-2">
                      <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-primary/10 text-primary">
                        <FolderTree className="h-4 w-4" />
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center justify-between gap-2">
                          <p className="min-w-0 break-words text-sm font-semibold leading-5">{cat.name}</p>
                          <p className="text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
                            {index + 1}
                          </p>
                        </div>
                        <p className="mt-1 break-words line-clamp-2 text-xs text-muted-foreground">
                          {cat.description || "No description provided"}
                        </p>
                      </div>
                    </div>

                    <Button
                      size="sm"
                      variant="destructive"
                      className="h-8 w-full rounded-lg text-[11px]"
                      onClick={() => handleDelete(cat.id)}
                    >
                      Delete
                    </Button>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
