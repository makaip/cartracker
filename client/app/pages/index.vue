<template>
    <UDashboardGroup>
        <!-- Sidebar -->
        <UDashboardSidebar resizable :default-size="20" :min-size="15" :max-size="50" class="border-r border-default">
            <UDashboardNavbar title="Sidebar" />

            <UModal title="Add Vehicle">
                <UButton label="Add Vehicle" color="neutral" variant="subtle" icon="i-lucide-plus"/>
                <template #body>
                    <UFileUpload label="Drop your image here" description="SVG, PNG, JPG or GIF (max. 2MB)" />
                </template>
            </UModal>
        </UDashboardSidebar>

        <!-- Main -->
        <UDashboardPanel grow>
            <UDashboardNavbar title="Camera">
                <template #trailing>
                    <UDropdownMenu :filter="{
                        icon: 'i-lucide-search'
                    }" :items="items" :content="{
                        align: 'start'
                    }" :ui="{
                        content: 'w-48'
                    }">
                        <UButton label="Open" icon="i-lucide-menu" color="neutral" variant="outline" />
                    </UDropdownMenu>
                </template>

                <template #right>
                    <UBadge label="4" variant="subtle" />
                </template>
            </UDashboardNavbar>

            <UDashboardToolbar>
                <template #left>
                    <Placeholder />
                </template>

                <template #right>
                    <Placeholder />
                </template>
            </UDashboardToolbar>

            <div class="p-6">
                <div class="h-150 rounded-lg bg-elevated flex items-center justify-center">
                    <Placeholder />
                </div>
            </div>
        </UDashboardPanel>

    </UDashboardGroup>


    <!-- modal   
  -->


</template>

<script setup lang="ts">
const Placeholder = defineComponent({
    name: 'Placeholder',
    template: `
    <div class="text-sm text-muted">
      Placeholder
    </div>
  `
})

import type { DropdownMenuItem } from '@nuxt/ui'

const items = ref<DropdownMenuItem[]>([
    {
        label: 'Auto Camera Track',
        icon: 'i-lucide-eye',
        onClick: () => {
            console.log('Auto Camera Track clicked')
        }
    },
])
</script>