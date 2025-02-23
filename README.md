<a name="top"></a>
<img src="static/images/banner.png">
[![language](https://img.shields.io/badge/language-python-green)]
[![course](https://img.shields.io/badge/Course-Learning Analytics-red)]
[![concepts](https://img.shields.io/badge/Recommendations-Content Based Filtering-yellow)]
[![university](https://img.shields.io/badge/University-Uni DUE-orange)]

<!-- ⭐ Star us on GitHub — it motivates us a lot! -->

[![Share](https://img.shields.io/badge/share-000000?logo=x&logoColor=white)](https://x.com/intent/tweet?text=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server%20%23OpenIDConnect%20%23Security%20%23Authentication)
[![Share](https://img.shields.io/badge/share-1877F2?logo=facebook&logoColor=white)](https://www.facebook.com/sharer/sharer.php?u=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0A66C2?logo=linkedin&logoColor=white)](https://www.linkedin.com/sharing/share-offsite/?url=https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-FF4500?logo=reddit&logoColor=white)](https://www.reddit.com/submit?title=Check%20out%20this%20project%20on%20GitHub:%20https://github.com/Abblix/Oidc.Server)
[![Share](https://img.shields.io/badge/share-0088CC?logo=telegram&logoColor=white)](https://t.me/share/url?url=https://github.com/Abblix/Oidc.Server&text=Check%20out%20this%20project%20on%20GitHub)

🔥 Why OIDC Server is the best choice for authentication — find out in our [presentation](https://resources.abblix.com/pdf/abblix-oidc-server-presentation-eng.pdf) 📑

## Table of Contents

- [About](#-about)
- [Certification](#-certification)
- [How to Build](#-how-to-build)
- [Documentation](#-documentation)
- [Feedback and Contributions](#-feedback-and-contributions)
- [License](#-license)
- [Contacts](#%EF%B8%8F-contacts)

## 🚀 About

**Abblix OIDC Server** is a .NET library designed to provide comprehensive support for OAuth2 and OpenID Connect on the server side. It adheres to high standards of flexibility, reusability, and reliability, utilizing well-known software design patterns, including modular and hexagonal architectures. These patterns ensure the following benefits:

- **Modularity**: Different parts of the library can function independently, enhancing the library's modularity and allowing for easier maintenance and updates.
- **Testability**: Improved separation of concerns makes the code more testable.
- **Maintainability**: Clear structure and separation facilitate better management of the codebase.

The library also supports Dependency Injection through the standard .NET DI container, aiding in the organization and management of code. Specifically tailored for seamless integration with ASP.NET WebApi, Abblix OIDC Server employs standard controller classes, binding, and routing mechanisms, simplifying the integration of OpenID Connect into your services.

## 🎓 Certification

[![OpenID_Foundation_Certification](https://resources.abblix.com/imgs/svg/abblix-oidc-server-openid-foundation-certification-mark.svg)](https://openid.net/certification/#OPENID-OP-P)

We are certified in all profiles. During the certification process, we skipped ZERO tests and received NO warnings. All **630** tests ![Passed](https://img.shields.io/badge/PASSED-brightgreen). We are extremely proud of this achievement. It reflects our overall approach to any endeavor. For more details, click the links ([Certified OpenID Providers & Profiles](https://openid.net/certification/#OPENID-OP-P), [Certified OpenID Providers for Logout Profiles](https://openid.net/certification/#OPENID-OP-LP)).

For convenience, the certification information is provided in the tables below:

### Regular Profiles

| OIDC Profile      | Response Types (links to official OpenID Foundation test results)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Tests   |
| :---------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| Basic OP          | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=FQHG2VEct7wwy)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             | 37      |
| Implicit OP       | [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=cQxDzZ2AF6kCd)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | 58      |
| Hybrid OP         | [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=3axSifHKQAsx7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    | 102     |
| Config OP         | [config](https://www.certification.openid.net/plan-detail.html?public=true&plan=MJLv3jEc7mjPa)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           | 1       |
| Dynamic OP        | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=spiRykedA0Mwt) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=hCPd3uy7p51qB) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=SWEcCmiFGq1I7) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=jalDvOupdIqzv) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=MYpjkY7SCvGGm) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=44V5kg4AZ8DCw) | 121     |
| Form Post OP      | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=mmLewPQ5CRI5Y) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=P9MbddCvNiOhV) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=m201EDuUSSGC4)                                                                                                                                                                                                                                                                                                                                | 197     |
| 3rd Party-Init OP | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=AZDaS1Bgcz05b) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=mV6GV7UKFChOK) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=tvdcmS6fHL5M9) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=APDFSOinLbRtE) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=Ra3nyzTuFrk3x) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=6rD5NRaBi2PKE) | 12      |
| **Total**         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **528** |

### Logout Profiles

| OIDC Profile     | Response Types (links to official OpenID Foundation test results)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Tests   |
| :--------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------ |
| RP-Initiated OP  | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=t3gEmJDtpGeBv) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=syvM21k6RaXZ4) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=0eADBDrm7CPoM) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=KMvHpgRYqQXQf) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=Oe06IYAYLq0Sg) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=xyZDiSE7Cm7ot) | 66      |
| Session OP       | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=14cK3zXIwOHZ1) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=L6Joz3e8tutYO) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=9pE1uRAUfsyQK) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=pizIIDc4VGrZO) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=MP0JEAuuLFWAr) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=mzm32Q7f2cT2n) | 12      |
| Front-Channel OP | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=3D5QcW9lq6H0Z) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=RtoGgLcrN0n6V) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=aba6tpe9d5CcU) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=ICPhbelPUUz4H) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=OYv7ccnbWElth) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=S0vzX58TaqBuC) | 12      |
| Back-Channel OP  | [code](https://www.certification.openid.net/plan-detail.html?public=true&plan=muYvBNgY6O90P) \| [code id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=uuGUH9zx9060p) \| [code id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=SQlr1myJDt2qb) \| [code token](https://www.certification.openid.net/plan-detail.html?public=true&plan=2A2MFQqpYNFz2) \| [id_token](https://www.certification.openid.net/plan-detail.html?public=true&plan=W7SL9FbkaqM5w) \| [id_token token](https://www.certification.openid.net/plan-detail.html?public=true&plan=ujWHBnctwGUvk) | 12      |
| **Total**        |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          | **102** |

## 📝 How to Build

To build the packages, follow these steps:

```shell
# Open a terminal (Command Prompt or PowerShell for Windows, Terminal for macOS or Linux)

# Ensure Git is installed
# Visit https://git-scm.com to download and install console Git if not already installed

# Clone the repository
git clone https://github.com/Abblix/Oidc.Server.git

# Navigate to the project directory
cd Oidc.Server

# Check if .NET SDK is installed
dotnet --version  # Check the installed version of .NET SDK
# Visit the official Microsoft website to install or update it if necessary

# Restore dependencies
dotnet restore

# Compile the project
dotnet build

```

## 📚 Documentation

### Getting Started

Explore the [Getting Started Guide](https://docs.abblix.com/docs/getting-started-guide).
In this guide, you will create a working solution step by step, building an OpenID Connect Provider using ASP.NET MVC and the Abblix OIDC Server solution.

To better understand the Abblix OIDC Server product, we recommend visiting our [Documentation](https://docs.abblix.com/docs) site. There, you will find useful information about the product and the OpenID Connect standard.

### Abblix OIDC Server Helper

The [**Abblix OIDC Server Helper**](https://chat.openai.com/g/g-1icXaNyOR-abblix-oidc-server-helper) is a specialized ChatGPT [![Abblix OIDC Server Helper](https://resources.abblix.com/imgs/svg/openai-black-15x15.svg)](https://chat.openai.com/g/g-1icXaNyOR-abblix-oidc-server-helper) designed to assist users and developers working with the Abblix OIDC Server. This tool is equipped with comprehensive documentation and various supporting documents, enabling it to provide precise and detailed assistance on a range of topics. Here are the key areas it can help with:

- **Integration**: Guidance on how to integrate the Abblix OIDC Server with your .NET project.
- **Configuration**: Best practices and settings for configuring the server.
- **Technical Support**: Troubleshooting common issues and implementing features.
- **Licensing and Pricing**: Detailed information on different licensing plans and costs.
- **Features and Benefits**: Overview of the server's features and their advantages.
- **Security and Compliance**: Ensuring your implementation is secure and compliant with standards.
- **Extending and Customizing**: Guidelines for extending the functionality of the server.

and much more.

This tool is ideal for both new users and experienced developers seeking specific technical details or support.

## 🤝 Feedback and Contributions

We've made every effort to implement all the main aspects of the OpenID protocol in the best possible way. However, the development journey doesn't end here, and your input is crucial for our continuous improvement.

> [!IMPORTANT]
> Whether you have feedback on features, have encountered any bugs, or have suggestions for enhancements, we're eager to hear from you. Your insights help us make the Abblix OIDC Server library more robust and user-friendly.

Please feel free to contribute by [submitting an issue](https://github.com/Abblix/Oidc.Server/issues) or [joining the discussions](https://github.com/orgs/Abblix/discussions). Each contribution helps us grow and improve.

We appreciate your support and look forward to making our product even better with your help!

## 📃 License

This product is distributed under a proprietary license. You can review the full license agreement at the following link: [Standard License Agreement for Abblix OIDC Server](https://resources.abblix.com/docs/eng/standard-license-agreement-abblix-oidc-server.pdf).

For non-commercial use, this product is available for free.

## 🗨️ Contacts

For more details about our products, services, or any general information regarding the Abblix OIDC Server, feel free to reach out to us. We are here to provide support and answer any questions you may have. Below are the best ways to contact our team:

- **Email**: Send us your inquiries or support requests at [support@abblix.com](mailto:support@abblix.com).
- **Website**: Visit the official Abblix OIDC Server page for more information: [Abblix OIDC Server](https://www.abblix.com/abblix-oidc-server).

Subscribe to our LinkedIn and Twitter:

[![LinkedIn](https://img.shields.io/badge/subscribe-white.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTIwLjQ0NyAyMC40NTJoLTMuNTU0di01LjU2OWMwLTEuMzI4LS4wMjctMy4wMzctMS44NTItMy4wMzctMS44NTMgMC0yLjEzNiAxLjQ0NS0yLjEzNiAyLjkzOXY1LjY2N0g5LjM1MVY5aDMuNDE0djEuNTYxaC4wNDZjLjQ3Ny0uOSAxLjYzNy0xLjg1IDMuMzctMS44NSAzLjYwMSAwIDQuMjY3IDIuMzcgNC4yNjcgNS40NTV2Ni4yODZ6TTUuMzM3IDcuNDMzYTIuMDYyIDIuMDYyIDAgMCAxLTIuMDYzLTIuMDY1IDIuMDY0IDIuMDY0IDAgMSAxIDIuMDYzIDIuMDY1em0xLjc4MiAxMy4wMTlIMy41NTVWOWgzLjU2NHYxMS40NTJ6TTIyLjIyNSAwSDEuNzcxQy43OTIgMCAwIC43NzQgMCAxLjcyOXYyMC41NDJDMCAyMy4yMjcuNzkyIDI0IDEuNzcxIDI0aDIwLjQ1MUMyMy4yIDI0IDI0IDIzLjIyNyAyNCAyMi4yNzFWMS43MjlDMjQgLjc3NCAyMy4yIDAgMjIuMjIyIDBoLjAwM3oiIGZpbGw9IiMwQTY2QzIiLz48cGF0aCBzdHlsZT0iZmlsbDojZmZmO3N0cm9rZS13aWR0aDouMDIwOTI0MSIgZD0iTTQuOTE3IDcuMzc3YTIuMDUyIDIuMDUyIDAgMCAxLS4yNC0zLjk0OWMxLjEyNS0uMzg0IDIuMzM5LjI3NCAyLjY1IDEuNDM3LjA2OC4yNS4wNjguNzY3LjAwMSAxLjAxYTIuMDg5IDIuMDg5IDAgMCAxLTEuNjIgMS41MSAyLjMzNCAyLjMzNCAwIDAgMS0uNzktLjAwOHoiLz48cGF0aCBzdHlsZT0iZmlsbDojZmZmO3N0cm9rZS13aWR0aDouMDIwOTI0MSIgZD0iTTQuOTE3IDcuMzc3YTIuMDU2IDIuMDU2IDAgMCAxLTEuNTItMi42NyAyLjA0NyAyLjA0NyAwIDAgMSAzLjQxOS0uNzU2Yy4yNC4yNTQuNDIuNTczLjUxMi45MDguMDY1LjI0LjA2NS43OCAwIDEuMDItLjA1MS4xODYtLjE5Ny41MDQtLjMuNjUyLS4wOS4xMzItLjMxLjM2Mi0uNDQzLjQ2NC0uNDYzLjM1Ny0xLjEuNTAzLTEuNjY4LjM4MlpNMy41NTcgMTQuNzJWOS4wMDhoMy41NTd2MTEuNDI0SDMuNTU3Wk05LjM1MyAxNC43MlY5LjAwOGgzLjQxMXYuNzg1YzAgLjYxNC4wMDUuNzg0LjAyNi43ODMuMDE0IDAgLjA3LS4wNzMuMTI0LS4xNjIuNTI0LS44NjUgMS41MDgtMS40NzggMi42NS0xLjY1LjI3NS0uMDQyIDEtLjA0NyAxLjMzMi0uMDA5Ljc5LjA5IDEuNDUxLjMxNiAxLjk0LjY2NC4yMi4xNTcuNTU3LjQ5My43MTQuNzEzLjQyLjU5Mi42OSAxLjQxMi44MDggMi40NjQuMDc0LjY2My4wODQgMS4yMTUuMDg1IDQuNTc4djMuMjU4aC0zLjUzNnYtMi45ODZjMC0yLjk3LS4wMS0zLjQ3NC0uMDc0LTMuOTA4LS4wOS0uNjA2LS4zMTQtMS4wODItLjYzNC0xLjM0Mi0uMzk1LS4zMjItMS4wMjktLjQzNy0xLjcwMy0uMzA5LS44NTguMTYzLTEuMzU1Ljc1LTEuNTIzIDEuNzk3LS4wNzYuNDcxLS4wODQuODQ1LS4wODQgMy44MzR2Mi45MTRIOS4zNTN6Ii8+PC9zdmc+)](https://www.linkedin.com/company/103540510/)
[![X](https://img.shields.io/badge/subscribe-white.svg?logo=data:image/svg%2bxml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PHBhdGggZD0iTTE4LjkwMSAxLjE1M2gzLjY4bC04LjA0IDkuMTlMMjQgMjIuODQ2aC03LjQwNmwtNS44LTcuNTg0LTYuNjM4IDcuNTg0SC40NzRsOC42LTkuODNMMCAxLjE1NGg3LjU5NGw1LjI0MyA2LjkzMlpNMTcuNjEgMjAuNjQ0aDIuMDM5TDYuNDg2IDMuMjRINC4yOThaIi8+PHBhdGggc3R5bGU9ImZpbGw6I2ZmZjtzdHJva2Utd2lkdGg6LjAyMDkyNDEiIGQ9Ik0xMS4wMzYgMTIuMDI4IDQuMzg3IDMuMzM0bC0uMDYtLjA4SDYuNDhsNi41MTYgOC42MTQgNi41NzUgOC42OTQuMDYuMDhoLTIuMDA2eiIvPjwvc3ZnPg==)](https://twitter.com/OIDCServer)

We look forward to assisting you and ensuring your experience with our products is successful and enjoyable!

[Back to top](#top)
