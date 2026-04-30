"use client"

import { useState, useEffect } from "react"
import axios from "axios"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { toast } from "sonner"

const API =
  `${process.env.REACT_APP_BACKEND_URL || "http://localhost:8000"}/api`

export default function InventoryActionModal({
  sku,
  mode,
  onSuccess
}) {

  const [lookupData, setLookupData] = useState(null)
  const [qtyInputs, setQtyInputs] = useState({})
  const [reason, setReason] = useState("")
  const [loading, setLoading] = useState(false)

  // ================= LOOKUP =================
  const lookupSku = async () => {
    try {

      const res = await axios.get(`${API}/inventory/lookup/${sku}`)

      setLookupData(res.data)

      const map = {}

      if (res.data.variants?.length > 0) {
        res.data.variants.forEach(v => {
          map[v.v_sku] = ""
        })
      } else {
        map[res.data.parent_sku] = ""
      }

      setQtyInputs(map)

    } catch (e) {
      toast.error("SKU not found")
    }
  }

  useEffect(() => {
    if (sku) lookupSku()
  }, [sku])

  // ================= QTY =================
  const handleQtyChange = (sku, value) => {
    setQtyInputs(prev => ({ ...prev, [sku]: value }))
  }

  // ================= SUBMIT =================
  const submitAction = async () => {

    try {

      setLoading(true)

      const reqs = Object.entries(qtyInputs)
        .filter(([_, q]) => Number(q) > 0)
        .map(([sku, q]) => {

          if (mode === "inward") {
            return axios.post(`${API}/inventory/material-inward/sku`, {
              sku,
              quantity: Number(q),
            })
          }

          return axios.post(`${API}/inventory/material-outward/sku`, {
            sku,
            quantity: Number(q),
            reason,
          })

        })

      if (!reqs.length) {
        toast.error("Enter quantity")
        return
      }

      await Promise.all(reqs)

      toast.success(
        mode === "inward"
          ? "Stock added successfully"
          : "Stock deducted successfully"
      )

      onSuccess?.()

    } catch (e) {
      toast.error(e.response?.data?.detail || "Failed")
    } finally {
      setLoading(false)
    }
  }

  if (!lookupData) return null

  const variants =
    lookupData.variants?.length > 0
      ? lookupData.variants
      : [
          {
            v_sku: lookupData.parent_sku,
            stock: lookupData.total_stock,
          },
        ]

  return (
    <div className="space-y-5">

      {/* PRODUCT HEADER */}
      <div className="border rounded-xl p-4 bg-muted/30">
        <div className="text-lg font-bold">
          {lookupData.product_name}
        </div>

        <div className="text-xs text-muted-foreground font-mono">
          Parent SKU: {lookupData.parent_sku}
        </div>

        <div className="text-sm mt-1 font-semibold">
          Total Stock: {lookupData.total_stock}
        </div>
      </div>

      {/* VARIANTS */}
      <div className="space-y-3 max-h-[350px] overflow-y-auto">

        {variants.map(v => (

          <div
            key={v.v_sku}
            className="border rounded-xl p-4 bg-background shadow-sm hover:bg-muted/20 transition"
          >

            <div className="flex justify-between items-center mb-2">

              <div className="flex flex-col">

                <span className="font-semibold text-sm">
                  {v.variant_name || "Standard"}
                </span>

                <span className="text-xs text-muted-foreground font-mono">
                  {v.v_sku}
                </span>

                <div className="flex gap-2 mt-1">

                  {v.color && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-blue-500/10 text-blue-600 font-semibold">
                      {v.color}
                    </span>
                  )}

                  {v.size && (
                    <span className="text-[10px] px-2 py-0.5 rounded bg-purple-500/10 text-purple-600 font-semibold">
                      {v.size}
                    </span>
                  )}

                </div>

              </div>

              <div className="text-right">

                <div className="text-xs text-muted-foreground">
                  Current Stock
                </div>

                <div className="font-bold text-sm text-green-600">
                  {v.stock}
                </div>

              </div>

            </div>

            {/* QTY INPUT */}
            <Input
              type="number"
              placeholder="Enter quantity"
              className="h-10"
              value={qtyInputs[v.v_sku] || ""}
              onChange={e =>
                handleQtyChange(v.v_sku, e.target.value)
              }
            />

          </div>

        ))}

      </div>

      {/* REASON */}
      {mode === "outward" && (
        <div className="space-y-2">
          <Label>Reason for deduction</Label>
          <Input
            placeholder="Damaged / Sold / Returned"
            value={reason}
            onChange={e => setReason(e.target.value)}
          />
        </div>
      )}

      {/* SUBMIT */}
      <Button
        className={`w-full h-11 text-sm font-semibold ${
          mode === "inward"
            ? "bg-green-600 hover:bg-green-700"
            : "bg-red-600 hover:bg-red-700"
        }`}
        onClick={submitAction}
        disabled={loading}
      >
        {loading
          ? "Processing..."
          : mode === "inward"
          ? "Add Stock"
          : "Deduct Stock"}
      </Button>

    </div>
  )
}