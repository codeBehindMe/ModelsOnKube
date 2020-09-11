variable "resource_region" {
  type = string
  default = "australia-southeast1"
}
variable "resource_zone" {
  type = string
  default = "australia-southeast1-c"
}

variable "small_node_type" {
  type = string
  default = "e2-medium"
  // FIXME: Bind a better size here
}

variable "medium_node_type" {
  type = string
  default = "e2-medium"
}

variable "large_node_type" {
  type = string
  default = "e2-medium"
  // FIXME: Bind a better size here
}
provider "google" {
  credentials = file("credentials.json")
  // FIXME: Make a credentials file
  project = ""
  // FIXME: Create a project for this
  region = var.resource_region
}

resource "google_container_cluster" "maas" {
  name = "maas"
  location = var.resource_zone
  initial_node_count = 1
  remove_default_node_pool = true
}

resource "google_container_node_pool" "multiplexers" {
  name = "multiplexers"
  location = var.resource_zone
  cluster = google_container_cluster.maas.name
  node_count = 3

  node_config {
    preemptible = false
    machine_type = var.small_node_type

    metadata = {
      disable-legacy-endpoints = true
      workload = "multiplexer"
      size = "small"
    }
  }
}

resource "google_container_node_pool" "small_executors" {
  name = "small_executors"
  location = var.resource_zone
  cluster = google_container_cluster.maas.name
  node_count = 3

  node_config {
    preemptible = true
    machine_type = var.small_node_type

    metadata = {
      disable-legacy-endpoints = true
      workload = "executor"
      size = "small"
    }
  }
}

resource "google_container_node_pool" "medium_executors" {
  name = "medium_executors"
  location = var.resource_zone
  cluster = google_container_cluster.maas.name
  node_count = 3

  node_config {
    preemptible = true
    machine_type = var.medium_node_type

    metadata = {
      disable-legacy-endpoints = true
      workload = "executor"
      size = "medium"
    }
  }
}

resource "google_container_node_pool" "large_executors" {
  name = "large_executors"
  location = var.resource_zone
  cluster = google_container_cluster.maas.name
  node_count = 3

  node_config {
    preemptible = true
    machine_type = var.large_node_type

    metadata = {
      disable-legacy-endpoints = true
      workload = "executor"
      size = "large"
    }
  }
}
