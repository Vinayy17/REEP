import requests
import sys
import json
from datetime import datetime

class InventoryAPITester:
    def __init__(self, base_url="https://order-dashboard-21.preview.emergentagent.com/api"):
        self.base_url = base_url
        self.token = None
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []
        self.created_ids = {
            'user_id': None,
            'category_id': None,
            'product_id': None,
            'customer_id': None,
            'invoice_id': None
        }

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
            print(f"✅ {name} - PASSED")
        else:
            print(f"❌ {name} - FAILED: {details}")
        
        self.test_results.append({
            'test': name,
            'status': 'PASSED' if success else 'FAILED',
            'details': details
        })

    def run_test(self, name, method, endpoint, expected_status, data=None, auth_required=True):
        """Run a single API test"""
        url = f"{self.base_url}/{endpoint}"
        headers = {'Content-Type': 'application/json'}
        
        if auth_required and self.token:
            headers['Authorization'] = f'Bearer {self.token}'

        print(f"\n🔍 Testing {name}...")
        print(f"   URL: {url}")
        
        try:
            if method == 'GET':
                response = requests.get(url, headers=headers, timeout=10)
            elif method == 'POST':
                response = requests.post(url, json=data, headers=headers, timeout=10)
            elif method == 'PUT':
                response = requests.put(url, json=data, headers=headers, timeout=10)
            elif method == 'DELETE':
                response = requests.delete(url, headers=headers, timeout=10)
            elif method == 'PATCH':
                response = requests.patch(url, headers=headers, timeout=10)

            success = response.status_code == expected_status
            
            if success:
                self.log_test(name, True)
                try:
                    return True, response.json()
                except:
                    return True, {}
            else:
                error_msg = f"Expected {expected_status}, got {response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f" - {error_detail}"
                except:
                    error_msg += f" - {response.text[:200]}"
                
                self.log_test(name, False, error_msg)
                return False, {}

        except Exception as e:
            self.log_test(name, False, f"Request failed: {str(e)}")
            return False, {}

    def test_user_registration(self):
        """Test user registration"""
        test_email = f"test_user_{datetime.now().strftime('%H%M%S')}@example.com"
        user_data = {
            "email": test_email,
            "password": "TestPass123!",
            "name": "Test User"
        }
        
        success, response = self.run_test(
            "User Registration",
            "POST",
            "auth/register",
            200,
            data=user_data,
            auth_required=False
        )
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            self.created_ids['user_id'] = response.get('user', {}).get('id')
            return True, test_email, "TestPass123!"
        return False, None, None

    def test_user_login(self, email, password):
        """Test user login"""
        login_data = {
            "email": email,
            "password": password
        }
        
        success, response = self.run_test(
            "User Login",
            "POST",
            "auth/login",
            200,
            data=login_data,
            auth_required=False
        )
        
        if success and 'access_token' in response:
            self.token = response['access_token']
            return True
        return False

    def test_categories_crud(self):
        """Test category CRUD operations"""
        # Create category
        category_data = {
            "name": "Test Electronics",
            "description": "Electronic devices and accessories"
        }
        
        success, response = self.run_test(
            "Create Category",
            "POST",
            "categories",
            200,
            data=category_data
        )
        
        if success and 'id' in response:
            self.created_ids['category_id'] = response['id']
        
        # Get categories
        self.run_test(
            "Get Categories",
            "GET",
            "categories",
            200
        )
        
        # Delete category (will do this later after products are tested)
        return success

    def test_products_crud(self):
        """Test product CRUD operations"""
        if not self.created_ids['category_id']:
            self.log_test("Product CRUD", False, "No category available for product creation")
            return False
            
        # Create product
        product_data = {
            "name": "Test Laptop",
            "description": "High-performance laptop for testing",
            "category_id": self.created_ids['category_id'],
            "price": 999.99,
            "stock": 50,
            "sku": f"LAPTOP-{datetime.now().strftime('%H%M%S')}",
            "image_url": "https://example.com/laptop.jpg"
        }
        
        success, response = self.run_test(
            "Create Product",
            "POST",
            "products",
            200,
            data=product_data
        )
        
        if success and 'id' in response:
            self.created_ids['product_id'] = response['id']
        
        # Get products
        self.run_test(
            "Get Products",
            "GET",
            "products",
            200
        )
        
        # Update product
        if self.created_ids['product_id']:
            updated_data = {**product_data, "price": 899.99, "stock": 45}
            self.run_test(
                "Update Product",
                "PUT",
                f"products/{self.created_ids['product_id']}",
                200,
                data=updated_data
            )
        
        return success

    def test_customers_crud(self):
        """Test customer CRUD operations"""
        # Create customer
        customer_data = {
            "name": "John Doe",
            "email": f"john.doe.{datetime.now().strftime('%H%M%S')}@example.com",
            "phone": "+1234567890",
            "address": "123 Test Street, Test City, TC 12345"
        }
        
        success, response = self.run_test(
            "Create Customer",
            "POST",
            "customers",
            200,
            data=customer_data
        )
        
        if success and 'id' in response:
            self.created_ids['customer_id'] = response['id']
        
        # Get customers
        self.run_test(
            "Get Customers",
            "GET",
            "customers",
            200
        )
        
        # Get specific customer
        if self.created_ids['customer_id']:
            self.run_test(
                "Get Customer by ID",
                "GET",
                f"customers/{self.created_ids['customer_id']}",
                200
            )
        
        return success

    def test_invoices_crud(self):
        """Test invoice CRUD operations"""
        if not self.created_ids['customer_id'] or not self.created_ids['product_id']:
            self.log_test("Invoice CRUD", False, "Missing customer or product for invoice creation")
            return False
            
        # Create invoice
        invoice_data = {
            "customer_id": self.created_ids['customer_id'],
            "items": [
                {
                    "product_id": self.created_ids['product_id'],
                    "product_name": "Test Laptop",
                    "quantity": 2,
                    "price": 899.99,
                    "total": 1799.98
                }
            ],
            "tax": 144.00,
            "discount": 50.00,
            "payment_status": "pending"
        }
        
        success, response = self.run_test(
            "Create Invoice",
            "POST",
            "invoices",
            200,
            data=invoice_data
        )
        
        if success and 'id' in response:
            self.created_ids['invoice_id'] = response['id']
        
        # Get invoices
        self.run_test(
            "Get Invoices",
            "GET",
            "invoices",
            200
        )
        
        # Get customer invoices
        if self.created_ids['customer_id']:
            self.run_test(
                "Get Customer Invoices",
                "GET",
                f"invoices/customer/{self.created_ids['customer_id']}",
                200
            )
        
        # Update invoice status
        if self.created_ids['invoice_id']:
            self.run_test(
                "Update Invoice Status",
                "PATCH",
                f"invoices/{self.created_ids['invoice_id']}/status?payment_status=paid",
                200
            )
        
        return success

    def test_dashboard_stats(self):
        """Test dashboard statistics"""
        success, response = self.run_test(
            "Get Dashboard Stats",
            "GET",
            "dashboard",
            200
        )
        
        if success:
            # Verify dashboard data structure
            required_fields = ['total_sales', 'total_orders', 'total_customers', 'low_stock_items', 'recent_invoices']
            missing_fields = [field for field in required_fields if field not in response]
            
            if missing_fields:
                self.log_test("Dashboard Data Structure", False, f"Missing fields: {missing_fields}")
            else:
                self.log_test("Dashboard Data Structure", True)
        
        return success

    def test_inventory_deduction(self):
        """Test inventory auto-deduction after invoice creation"""
        if not self.created_ids['product_id']:
            self.log_test("Inventory Deduction", False, "No product available for testing")
            return False
            
        # Get current stock
        success, products = self.run_test(
            "Get Products for Stock Check",
            "GET",
            "products",
            200
        )
        
        if success:
            product = next((p for p in products if p['id'] == self.created_ids['product_id']), None)
            if product:
                # Stock should be reduced from original 50 to 45 (updated) to 43 (after invoice of 2 items)
                expected_stock = 43  # 45 - 2 from invoice
                actual_stock = product['stock']
                
                if actual_stock == expected_stock:
                    self.log_test("Inventory Auto-Deduction", True)
                    return True
                else:
                    self.log_test("Inventory Auto-Deduction", False, 
                                f"Expected stock: {expected_stock}, Actual: {actual_stock}")
            else:
                self.log_test("Inventory Auto-Deduction", False, "Product not found")
        
        return False

    def cleanup_test_data(self):
        """Clean up created test data"""
        print("\n🧹 Cleaning up test data...")
        
        # Delete product first (depends on category)
        if self.created_ids['product_id']:
            self.run_test(
                "Delete Product",
                "DELETE",
                f"products/{self.created_ids['product_id']}",
                200
            )
        
        # Delete category
        if self.created_ids['category_id']:
            self.run_test(
                "Delete Category",
                "DELETE",
                f"categories/{self.created_ids['category_id']}",
                200
            )

    def run_all_tests(self):
        """Run all API tests"""
        print("🚀 Starting Inventory Management API Tests")
        print(f"Base URL: {self.base_url}")
        print("=" * 60)
        
        # Test authentication
        reg_success, email, password = self.test_user_registration()
        if not reg_success:
            print("❌ Registration failed, stopping tests")
            return self.generate_report()
        
        login_success = self.test_user_login(email, password)
        if not login_success:
            print("❌ Login failed, stopping tests")
            return self.generate_report()
        
        # Test CRUD operations
        self.test_categories_crud()
        self.test_products_crud()
        self.test_customers_crud()
        self.test_invoices_crud()
        
        # Test dashboard and inventory features
        self.test_dashboard_stats()
        self.test_inventory_deduction()
        
        # Cleanup
        self.cleanup_test_data()
        
        return self.generate_report()

    def generate_report(self):
        """Generate test report"""
        print("\n" + "=" * 60)
        print("📊 TEST RESULTS SUMMARY")
        print("=" * 60)
        print(f"Tests Run: {self.tests_run}")
        print(f"Tests Passed: {self.tests_passed}")
        print(f"Tests Failed: {self.tests_run - self.tests_passed}")
        print(f"Success Rate: {(self.tests_passed/self.tests_run*100):.1f}%" if self.tests_run > 0 else "0%")
        
        if self.tests_passed < self.tests_run:
            print("\n❌ FAILED TESTS:")
            for result in self.test_results:
                if result['status'] == 'FAILED':
                    print(f"  - {result['test']}: {result['details']}")
        
        return {
            'total_tests': self.tests_run,
            'passed_tests': self.tests_passed,
            'failed_tests': self.tests_run - self.tests_passed,
            'success_rate': (self.tests_passed/self.tests_run*100) if self.tests_run > 0 else 0,
            'test_results': self.test_results
        }

def main():
    tester = InventoryAPITester()
    report = tester.run_all_tests()
    
    # Return appropriate exit code
    return 0 if report['failed_tests'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())