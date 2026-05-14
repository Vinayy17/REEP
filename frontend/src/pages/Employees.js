"use client"

import { useEffect, useState } from "react"
import axios from "axios"
import { useLocation, useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { toast } from "sonner"
import EntityCard from "@/components/people/EntityCard"
import StatsCardGrid from "@/components/people/StatsCardGrid"
import {
  Eye,
  Edit,
  Plus,
  Search,
  Trash2,
  Users,
  Wallet,
} from "lucide-react"

const API = `${process.env.REACT_APP_BACKEND_URL}/api`

const emptyEmployee = {
  name: "",
  role: "",
  salary: "",
}

const formatMoney = (value) =>
  new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 2,
    minimumFractionDigits: 2,
  }).format(Number(value || 0))

const formatCurrency = (value) => `\u20B9\u00A0${formatMoney(value)}`

const formatDate = (value) => {
  if (!value) return "-"

  try {
    return new Date(value).toLocaleDateString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
    })
  } catch {
    return value
  }
}

export default function Employees() {
  const location = useLocation()
  const navigate = useNavigate()

  const [employees, setEmployees] = useState([])
  const [searchTerm, setSearchTerm] = useState("")
  const [dialogOpen, setDialogOpen] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [editingEmployee, setEditingEmployee] = useState(null)
  const [viewEmployee, setViewEmployee] = useState(null)
  const [employeeForm, setEmployeeForm] = useState(emptyEmployee)
  const [payDialogOpen, setPayDialogOpen] = useState(false)
  const [paySubmitting, setPaySubmitting] = useState(false)
  const [payEmployee, setPayEmployee] = useState(null)
  const [payAmount, setPayAmount] = useState("")
  const [payMode, setPayMode] = useState("upi")
  const [payDate, setPayDate] = useState(() => new Date().toISOString().slice(0, 10))
  const [payNotes, setPayNotes] = useState("")

  const tokenHeaders = () => ({
    headers: {
      Authorization: `Bearer ${localStorage.getItem("token")}`,
    },
  })

  const resetEmployeeForm = () => {
    setEditingEmployee(null)
    setEmployeeForm(emptyEmployee)
  }

  const resetPayDialog = () => {
    setPayDialogOpen(false)
    setPaySubmitting(false)
    setPayEmployee(null)
    setPayAmount("")
    setPayMode("upi")
    setPayDate(new Date().toISOString().slice(0, 10))
    setPayNotes("")
  }

  const fetchEmployees = async () => {
    try {
      const res = await axios.get(`${API}/employees`, tokenHeaders())
      const data = Array.isArray(res.data) ? res.data : []
      setEmployees(data)
      return data
    } catch {
      toast.error("Failed to load employees")
      return []
    }
  }

  useEffect(() => {
    fetchEmployees()
  }, [])

  useEffect(() => {
    const params = new URLSearchParams(location.search)
    if (params.get("action") === "add") {
      setDialogOpen(true)
      params.delete("action")
      navigate(
        {
          pathname: "/employees",
          search: params.toString() ? `?${params.toString()}` : "",
        },
        { replace: true }
      )
    }
  }, [location.search, navigate])

  const openEditDialog = (employee) => {
    setEditingEmployee(employee)
    setEmployeeForm({
      name: employee.name || "",
      role: employee.role || "",
      salary: employee.salary ?? "",
    })
    setDialogOpen(true)
  }

  const openViewDialog = (employee) => {
    setViewEmployee(employee)
  }

  const submitEmployee = async (e) => {
    e.preventDefault()
    setSubmitting(true)

    try {
      const payload = {
        name: employeeForm.name.trim(),
        role: employeeForm.role.trim() || null,
        salary: Number(employeeForm.salary || 0),
      }

      if (!payload.name) {
        toast.error("Employee name is required")
        return
      }

      if (payload.salary <= 0) {
        toast.error("Salary must be greater than zero")
        return
      }

      if (editingEmployee) {
        await axios.put(`${API}/employees/${editingEmployee.id}`, payload, tokenHeaders())
        toast.success("Employee updated successfully")
      } else {
        await axios.post(`${API}/employees`, payload, tokenHeaders())
        toast.success("Employee added successfully")
      }

      await fetchEmployees()
      setDialogOpen(false)
      resetEmployeeForm()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to save employee")
    } finally {
      setSubmitting(false)
    }
  }

  const deleteEmployee = async (employee) => {
    const confirmed = window.confirm(`Delete employee ${employee.name}?`)
    if (!confirmed) return

    try {
      await axios.delete(`${API}/employees/${employee.id}`, tokenHeaders())
      toast.success("Employee deleted successfully")
      await fetchEmployees()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to delete employee")
    }
  }

  const openPayDialog = (employee) => {
    const pendingAmount = Number(employee?.pending_amount || 0)
    if (pendingAmount <= 0) {
      toast.error("No pending salary to pay for this employee")
      return
    }

    setPayEmployee(employee)
    setPayAmount(String(pendingAmount))
    setPayMode("upi")
    setPayDate(new Date().toISOString().slice(0, 10))
    setPayNotes("")
    setPayDialogOpen(true)
  }

  const submitEmployeePayment = async (event) => {
    event.preventDefault()

    if (!payEmployee) return

    const amount = Number(payAmount || 0)
    if (amount <= 0) {
      toast.error("Payment amount must be greater than zero")
      return
    }

    setPaySubmitting(true)

    try {
      await axios.post(
        `${API}/employees/${payEmployee.id}/salary-payments/pay`,
        {
          amount,
          payment_mode: payMode,
          payment_date: payDate || null,
          notes: payNotes || null,
        },
        tokenHeaders()
      )

      toast.success("Employee salary paid successfully")
      await fetchEmployees()
      resetPayDialog()
    } catch (error) {
      toast.error(error?.response?.data?.detail || "Failed to pay employee salary")
    } finally {
      setPaySubmitting(false)
    }
  }

  const filteredEmployees = employees
    .filter((employee) => {
      const query = searchTerm.trim().toLowerCase()
      if (!query) return true

      return [employee.name, employee.role, employee.salary_period]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    })
    .sort((a, b) => String(a.name || "").localeCompare(String(b.name || "")))

  const employeeStats = [
    {
      key: "salary-total",
      label: "Total Salary",
      value: formatCurrency(
        filteredEmployees.reduce((sum, employee) => sum + Number(employee.salary || 0), 0)
      ),
      tone: "info",
      indicatorTone: "info",
    },
    {
      key: "salary-paid",
      label: "Paid",
      value: formatCurrency(
        filteredEmployees.reduce((sum, employee) => sum + Number(employee.paid_amount || 0), 0)
      ),
      tone: "success",
      indicatorTone: "success",
    },
    {
      key: "salary-pending",
      label: "Pending",
      value: formatCurrency(
        filteredEmployees.reduce((sum, employee) => sum + Number(employee.pending_amount || 0), 0)
      ),
      tone: "warning",
      indicatorTone: "warning",
    },
  ]

  return (
    <div className="mx-auto max-w-7xl space-y-4 pb-4">
      <div className="space-y-4 px-4 pt-4 md:px-0 md:pt-0">
        <Dialog
          open={Boolean(viewEmployee)}
          onOpenChange={(next) => {
            if (!next) {
              setViewEmployee(null)
            }
          }}
        >
          <DialogContent className="max-h-[92vh] overflow-y-auto rounded-[24px] border-white/8 bg-[linear-gradient(180deg,rgba(6,11,26,0.96),rgba(6,11,26,0.9))] text-white backdrop-blur-xl sm:max-w-xl">
            <DialogHeader>
              <DialogTitle>Employee Details</DialogTitle>
            </DialogHeader>

            {viewEmployee ? (
              <div className="space-y-4">
                <div className="rounded-[20px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.035),rgba(255,255,255,0.015))] p-4 shadow-[0_18px_36px_rgba(2,6,23,0.1)] backdrop-blur-xl">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
                    <div className="space-y-1.5">
                      <p className="text-lg font-semibold text-white">{viewEmployee.name}</p>
                      <p className="text-sm text-slate-300">{viewEmployee.role || "No role added"}</p>
                      <p className="text-sm text-slate-400">{viewEmployee.salary_period || "-"}</p>
                      <p className="text-xs text-slate-500">Created {formatDate(viewEmployee.created_at)}</p>
                    </div>

                    <div className="grid min-w-0 grid-cols-3 gap-2 sm:min-w-[22rem]">
                      <EmployeeStatTile label="Salary" value={formatCurrency(viewEmployee.salary)} />
                      <EmployeeStatTile label="Paid" value={formatCurrency(viewEmployee.paid_amount)} tone="success" />
                      <EmployeeStatTile label="Pending" value={formatCurrency(viewEmployee.pending_amount)} tone="warning" />
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <InfoPanel title="Current Month Record">
                    <InfoLine label="Payment Date" value={formatDate(viewEmployee.open_salary_record?.payment_date)} />
                    <InfoLine label="Mode" value={viewEmployee.open_salary_record?.payment_mode ? String(viewEmployee.open_salary_record.payment_mode).toUpperCase() : "-"} />
                    <InfoLine label="Pending" value={formatCurrency(viewEmployee.open_salary_record?.pending_amount ?? viewEmployee.pending_amount)} />
                  </InfoPanel>

                  <InfoPanel title="Notes">
                    <p className="text-sm leading-6 text-slate-300">
                      {viewEmployee.open_salary_record?.notes || "No notes added for this salary record."}
                    </p>
                  </InfoPanel>
                </div>

                <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                  <Button
                    type="button"
                    variant="outline"
                    className="h-11 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white"
                    onClick={() => setViewEmployee(null)}
                  >
                    Close
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    className="h-11 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white"
                    onClick={() => {
                      setViewEmployee(null)
                      openEditDialog(viewEmployee)
                    }}
                  >
                    Edit
                  </Button>
                  <Button
                    type="button"
                    className="h-11 rounded-xl bg-indigo-500 text-white hover:bg-indigo-400"
                    onClick={() => {
                      setViewEmployee(null)
                      openPayDialog(viewEmployee)
                    }}
                    disabled={Number(viewEmployee.pending_amount || 0) <= 0}
                  >
                    Pay Employee
                  </Button>
                </div>
              </div>
            ) : null}
          </DialogContent>
        </Dialog>

        <Dialog
          open={dialogOpen}
          onOpenChange={(next) => {
            setDialogOpen(next)
            if (!next) {
              resetEmployeeForm()
            }
          }}
        >
          <DialogContent className="max-h-[92vh] overflow-y-auto border-white/10 bg-slate-950 text-white sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>{editingEmployee ? "Edit Employee" : "Add Employee"}</DialogTitle>
            </DialogHeader>

            <form className="space-y-4" onSubmit={submitEmployee}>
              <div className="space-y-2">
                <Label htmlFor="employee-name">Employee Name</Label>
                <Input
                  id="employee-name"
                  className="border-white/10 bg-white/5 text-white"
                  value={employeeForm.name}
                  onChange={(e) => setEmployeeForm((prev) => ({ ...prev, name: e.target.value }))}
                  placeholder="Enter employee name"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="employee-role">Role</Label>
                <Input
                  id="employee-role"
                  className="border-white/10 bg-white/5 text-white"
                  value={employeeForm.role}
                  onChange={(e) => setEmployeeForm((prev) => ({ ...prev, role: e.target.value }))}
                  placeholder="Sales, Billing, Warehouse..."
                />
              </div>

              <div className="space-y-2">
                <Label htmlFor="employee-salary">Monthly Salary</Label>
                <Input
                  id="employee-salary"
                  type="number"
                  min="0"
                  step="0.01"
                  className="border-white/10 bg-white/5 text-white"
                  value={employeeForm.salary}
                  onChange={(e) => setEmployeeForm((prev) => ({ ...prev, salary: e.target.value }))}
                  placeholder="0.00"
                  required
                />
              </div>

              <Button
                type="submit"
                className="h-12 w-full rounded-2xl bg-indigo-500 text-white hover:bg-indigo-400"
                disabled={submitting}
              >
                {submitting
                  ? editingEmployee
                    ? "Updating..."
                    : "Saving..."
                  : editingEmployee
                    ? "Update Employee"
                    : "Add Employee"}
              </Button>
            </form>
          </DialogContent>
        </Dialog>

        <Dialog
          open={payDialogOpen}
          onOpenChange={(next) => {
            if (!next) {
              resetPayDialog()
              return
            }

            setPayDialogOpen(true)
          }}
        >
          <DialogContent className="max-h-[92vh] overflow-y-auto border-white/10 bg-slate-950 text-white sm:max-w-lg">
            <DialogHeader>
              <DialogTitle>Pay Employee</DialogTitle>
            </DialogHeader>

            <form className="space-y-4" onSubmit={submitEmployeePayment}>
              <div className="rounded-2xl border border-white/10 bg-white/[0.04] p-3">
                <p className="text-sm font-semibold text-white">{payEmployee?.name || "-"}</p>
                <p className="mt-1 text-xs text-slate-400">{payEmployee?.role || "No role added"}</p>
                <p className="mt-1 text-xs text-amber-300">
                  Pending Salary: {formatCurrency(payEmployee?.pending_amount)}
                </p>
              </div>

              <div className="space-y-2">
                <Label>Payment Amount</Label>
                <Input
                  type="number"
                  min="0"
                  step="0.01"
                  className="border-white/10 bg-white/5 text-white"
                  value={payAmount}
                  onChange={(e) => setPayAmount(e.target.value)}
                  placeholder="0.00"
                  required
                />
              </div>

              <div className="space-y-2">
                <Label>Payment Mode</Label>
                <Select value={payMode} onValueChange={setPayMode}>
                  <SelectTrigger className="border-white/10 bg-white/5 text-white">
                    <SelectValue placeholder="Select payment mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="upi">UPI</SelectItem>
                    <SelectItem value="cash">Cash</SelectItem>
                    <SelectItem value="bank">Bank</SelectItem>
                    <SelectItem value="cheque">Cheque</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-2">
                <Label>Payment Date</Label>
                <Input
                  type="date"
                  className="border-white/10 bg-white/5 text-white"
                  value={payDate}
                  onChange={(e) => setPayDate(e.target.value)}
                />
              </div>

              <div className="space-y-2">
                <Label>Notes</Label>
                <Input
                  className="border-white/10 bg-white/5 text-white"
                  value={payNotes}
                  onChange={(e) => setPayNotes(e.target.value)}
                  placeholder="Optional notes"
                />
              </div>

              <Button
                type="submit"
                className="h-12 w-full rounded-2xl bg-indigo-500 text-white hover:bg-indigo-400"
                disabled={paySubmitting}
              >
                {paySubmitting ? "Paying..." : "Pay Employee"}
              </Button>
            </form>
          </DialogContent>
        </Dialog>

        <div className="flex flex-col gap-3 md:flex-row md:items-center">
          <div className="relative flex-1">
            <Search className="absolute left-4 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
            <Input
              placeholder="Search employee by name, role, or month..."
              className="h-12 rounded-xl border-white/8 bg-white/[0.02] pl-11 pr-4 text-sm text-white shadow-[0_14px_30px_rgba(2,6,23,0.08)] backdrop-blur-xl placeholder:text-slate-400"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <Button
            type="button"
            className="hidden h-12 rounded-xl bg-indigo-500 px-5 text-sm font-semibold text-white shadow-[0_16px_32px_rgba(99,102,241,0.24)] hover:bg-indigo-400 md:inline-flex"
            onClick={() => setDialogOpen(true)}
          >
            <Plus className="mr-2 h-4 w-4" />
            Add Employee
          </Button>
        </div>

        <StatsCardGrid items={employeeStats} className="grid-cols-3 md:grid-cols-3 lg:grid-cols-3" />
      </div>

      <div className="space-y-2.5">
        <div className="space-y-2.5 md:hidden">
            {filteredEmployees.length === 0 && (
              <div className="rounded-3xl border border-dashed border-white/10 bg-white/[0.03] px-6 py-12 text-center text-sm text-slate-400">
                No employees found
              </div>
            )}

            {filteredEmployees.map((employee) => (
              <EntityCard
                key={employee.id}
                icon={<Users className="h-4 w-4" />}
                title={employee.name}
                subtitle={employee.role || "No role added"}
                metaLines={[
                  employee.salary_period || "No salary period",
                  Number(employee.pending_amount || 0) > 0
                    ? `Pending Salary: ${formatCurrency(employee.pending_amount)}`
                    : "Salary settled",
                ]}
                amount={formatCurrency(employee.salary)}
                amountTone="success"
                status={
                  Number(employee.pending_amount || 0) > 0
                    ? { label: "Pending", tone: "warning" }
                    : { label: "Paid", tone: "success" }
                }
                breakdown={[
                  { key: "salary", label: "Salary", value: formatCurrency(employee.salary) },
                  {
                    key: "paid",
                    label: "Paid",
                    value: formatCurrency(employee.paid_amount),
                    tone: "success",
                  },
                  {
                    key: "pending",
                    label: "Pending",
                    value: formatCurrency(employee.pending_amount),
                    tone: "warning",
                  },
                ]}
                actions={[
                  {
                    key: "view",
                    label: "View",
                    tone: "secondary",
                    onClick: () => openViewDialog(employee),
                  },
                  {
                    key: "pay",
                    label: "Pay",
                    tone: "success",
                    onClick: () => openPayDialog(employee),
                    disabled: Number(employee.pending_amount || 0) <= 0,
                  },
                  {
                    key: "edit",
                    label: "Edit",
                    tone: "secondary",
                    onClick: () => openEditDialog(employee),
                  },
                  {
                    key: "delete",
                    label: "Delete",
                    tone: "danger",
                    onClick: () => deleteEmployee(employee),
                  },
                ]}
              />
            ))}
        </div>

        <div className="hidden overflow-hidden rounded-[20px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.02),rgba(255,255,255,0.008))] shadow-[0_18px_40px_rgba(2,6,23,0.1)] backdrop-blur-xl md:block">
          <Table>
            <TableHeader className="bg-white/[0.025]">
              <TableRow className="border-white/10">
                <TableHead className="text-slate-300">Employee</TableHead>
                <TableHead className="text-slate-300">Role</TableHead>
                <TableHead className="text-slate-300">Monthly Salary</TableHead>
                <TableHead className="text-slate-300">Paid</TableHead>
                <TableHead className="text-slate-300">Pending</TableHead>
                <TableHead className="text-slate-300">Period</TableHead>
                <TableHead className="text-slate-300">Created</TableHead>
                <TableHead className="text-right text-slate-300">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {filteredEmployees.length === 0 ? (
                <TableRow className="border-white/10">
                  <TableCell colSpan={8} className="py-10 text-center text-slate-400">
                    No employees found
                  </TableCell>
                </TableRow>
              ) : (
                filteredEmployees.map((employee) => (
                  <TableRow key={employee.id} className="border-white/10 text-slate-100">
                    <TableCell className="font-medium text-white">{employee.name}</TableCell>
                    <TableCell>{employee.role || "-"}</TableCell>
                    <TableCell>{formatCurrency(employee.salary)}</TableCell>
                    <TableCell className="text-emerald-300">{formatCurrency(employee.paid_amount)}</TableCell>
                    <TableCell className="text-amber-300">{formatCurrency(employee.pending_amount)}</TableCell>
                    <TableCell>{employee.salary_period}</TableCell>
                    <TableCell>{formatDate(employee.created_at)}</TableCell>
                    <TableCell>
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-9 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white"
                          onClick={() => openViewDialog(employee)}
                        >
                          <Eye className="mr-2 h-4 w-4" />
                          View
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-9 rounded-xl border-emerald-500/20 bg-emerald-500/10 text-emerald-200 hover:bg-emerald-500/20 hover:text-white"
                          onClick={() => openPayDialog(employee)}
                          disabled={Number(employee.pending_amount || 0) <= 0}
                        >
                          <Wallet className="mr-2 h-4 w-4" />
                          Pay
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-9 rounded-xl border-white/10 bg-transparent text-white hover:bg-white/5 hover:text-white"
                          onClick={() => openEditDialog(employee)}
                        >
                          <Edit className="mr-2 h-4 w-4" />
                          Edit
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="h-9 rounded-xl border-rose-500/20 bg-rose-500/10 text-rose-200 hover:bg-rose-500/20 hover:text-white"
                          onClick={() => deleteEmployee(employee)}
                        >
                          <Trash2 className="mr-2 h-4 w-4" />
                          Delete
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))
              )}
            </TableBody>
          </Table>
        </div>
      </div>
    </div>
  )
}

function EmployeeStatTile({ label, value, tone = "default" }) {
  const toneClasses = {
    default: "bg-white/[0.035] text-white",
    success: "bg-emerald-500/[0.12] text-emerald-300",
    warning: "bg-amber-500/[0.12] text-amber-200",
  }

  return (
    <div className={`rounded-[16px] border border-white/8 px-3 py-2 backdrop-blur-xl ${toneClasses[tone] || toneClasses.default}`}>
      <p className="text-[10px] uppercase tracking-[0.18em] text-slate-400">{label}</p>
      <p className="mt-1 text-sm font-semibold">{value}</p>
    </div>
  )
}

function InfoPanel({ title, children }) {
  return (
    <div className="rounded-[20px] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.03),rgba(255,255,255,0.012))] p-4 shadow-[0_18px_36px_rgba(2,6,23,0.08)] backdrop-blur-xl">
      <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-400">{title}</p>
      <div className="mt-3 space-y-2">{children}</div>
    </div>
  )
}

function InfoLine({ label, value }) {
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <span className="text-slate-400">{label}</span>
      <span className="text-right font-medium text-white">{value}</span>
    </div>
  )
}
